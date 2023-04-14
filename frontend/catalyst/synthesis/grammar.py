""" This module defines the abstract syntax tree structures for the output
language. The `Program` dataclass is supposed to be top-level node of the AST.
"""

from typing import (Any, List, Tuple, Optional, Dict, Union, NoReturn, Set, Callable)
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from functools import reduce
from jax import Array as JaxArray


@dataclass
class POI:
    """ Point Of Insertion. By convention, we allow inserting new
    statements strictly at the end of the list of already existing ones. """
    stmts:List["Stmt"]
    expr:"Expr"

    def __init__(self, stmts=None, expr=None):
        self.stmts = stmts if stmts is not None else []
        self.expr = expr if expr is not None else NoneExpr()

    def __hash__(self):
        return hash((tuple(self.stmts), self.expr))

    @classmethod
    def fromExpr(cls, e:"Expr") -> "POI":
        return POI([],e)

    @classmethod
    def fE(cls, *args, **kwargs) -> "POI":
        """ Alias """
        return cls.fromExpr(*args, **kwargs)



@dataclass(frozen=True)
class VName:
    """ Alias for strings representing variable names """
    val: str

    def __lt__(self, other):
        return self.val < other.val

@dataclass(frozen=True)
class FName:
    """ Alias for strings representing function names """
    val: str

    def __lt__(self, other):
        return self.val < other.val

Expr = Union["VRefExpr","FCallExpr", "ConstExpr", "NoneExpr", "CondExpr",
             "ForLoopExpr", "WhileLoopExpr" ]

def isinstance_expr(e:Any) -> bool:
    """Workaround for a TypeError, which is probably a Python bug."""
    return isinstance(e, (VRefExpr,FCallExpr, ConstExpr, NoneExpr, CondExpr,
                          ForLoopExpr, WhileLoopExpr))

@dataclass(frozen=True)
class VRefExpr:
    """ Expression - reference to a variable """
    vname: Union[FName, VName]

@dataclass(frozen=True)
class ConstExpr:
    """ Expression - constant """
    val: Union[bool, int, float, complex, JaxArray]

@dataclass(frozen=True)
class NoneExpr:
    """ Alias for None """
    pass

class ControlFlowStyle(Enum):
    Python = 0
    Catalyst = 1
    JAX = 2

@dataclass(frozen=True)
class CondExpr:
    """ Expression - conditional """
    cond: Expr
    trueBranch: POI
    falseBranch: Optional[POI]
    style: ControlFlowStyle

@dataclass(frozen=True)
class ForLoopExpr:
    """ Expression - for loop """
    loopvar: VName
    lbound: Expr
    ubound: Expr
    body: POI
    style: ControlFlowStyle
    statevar: Optional[VName] = None # TODO: Auto-generate this name in `pprint`

@dataclass(frozen=True)
class WhileLoopExpr:
    """ Expression - while loop """
    loopvar: VName
    cond: Expr
    body: POI
    style: ControlFlowStyle

@dataclass(frozen=True)
class FCallExpr:
    """ Expression - calling a callable """
    expr: Union[VRefExpr, CondExpr, ForLoopExpr, WhileLoopExpr]
    args: List[Expr]

    def __hash__(self):
        return hash((self.expr, tuple(self.args)))


trueExpr = ConstExpr(True)
falseExpr = ConstExpr(False)

def lessExpr(a,b) -> FCallExpr:
    return FCallExpr(VRefExpr(FName('<')),[a,b])

def addExpr(a,b) -> FCallExpr:
    return FCallExpr(VRefExpr(FName('+')),[a,b])


Stmt = Union["AssignStmt", "FDefStmt", "RetStmt"]

def isinstance_stmt(s:Any) -> bool:
    """Workaround for a TypeError, which is probably a Python bug."""
    return isinstance(s, (AssignStmt, FDefStmt, RetStmt))

@dataclass(frozen=True)
class AssignStmt:
    """ Statement - variable assignemnt or a function call """
    vname: Optional[VName]
    expr: Expr

    @classmethod
    def fE(cls, e:Expr) -> "AssignStmt":
        return AssignStmt(None, e)

@dataclass(frozen=True)
class FDefStmt:
    """ Statement - function declaration """
    fname: FName
    args: List[VName]
    body: POI
    qnode_wires: Optional[int] = None
    qnode_device: Optional[str] = None
    qjit: bool = False

@dataclass(frozen=True)
class RetStmt:
    """ Statement - return """
    expr: Optional[Expr]

Program = FDefStmt
""" Top-level program is a function declaration """


def assert_never(x: Any) -> NoReturn:
    raise RuntimeError("Unhandled type: {}".format(type(x).__name__))


@dataclass
class Signature:
    args:List[str]
    ret:str


def signature(x: Union[FDefStmt, Expr]) -> Optional[Signature]:
    """Return the signature: the "names" list of arguments and the return "type", or
    None for non-callable expressions. Currently, we don't use true type system.
    Note that loops and conditionals are callables in this language. """
    if isinstance(x, FDefStmt):
        return Signature(['*']*len(x.args), "*")
    elif isinstance(x, ForLoopExpr):
        return Signature(["*"], "*")
    elif isinstance(x, WhileLoopExpr):
        return Signature(["*"], "*")
    elif isinstance(x, CondExpr):
        return Signature([], "*")
    else:
        return None


def innerdefs1(e: Expr) -> Set[VRefExpr]:
    """ Return immediate inner variables """
    if isinstance(e, ForLoopExpr):
        return set([VRefExpr(e.loopvar)])
    elif isinstance(e, WhileLoopExpr):
        return set([VRefExpr(e.loopvar)])
    else:
        return set()


def bind(a:POI, b:POI, expr:Expr) -> POI:
    return POI(a.stmts + b.stmts, expr)


def bindUnary(value:POI, function:POI) -> Optional[POI]:
    s = signature(function.expr)
    if (s is None) or (len(s.args) != 1):
        return None
    return bind(value, function, FCallExpr(function.expr, [value.expr]))


def saturate_expr(e:Expr, arg:Expr) -> Expr:
    s = signature(e)
    return FCallExpr(e, [arg for _ in s.args]) if s else e


Acc = Any

def reduce_expr(e:Expr, f:Callable[[Expr,Acc],Acc], acc:Acc) -> Acc:
    def _down(subexprs):
        return reduce(lambda acc,se: reduce_expr(se,f,acc), subexprs, f(e,acc))
    if isinstance(e, FCallExpr):
        return _down(e.args)
    elif isinstance(e, CondExpr):
        return _down([e.trueBranch.expr]+([e.falseBranch.expr] if e.falseBranch else []))
    elif isinstance(e, ForLoopExpr):
        return _down([e.lbound, e.ubound, e.body.expr])
    elif isinstance(e, WhileLoopExpr):
        return _down([e.cond, e.body.expr])
    elif isinstance(e, (NoneExpr, VRefExpr, ConstExpr)):
        return _down([])
    else:
        assert_never(e)





