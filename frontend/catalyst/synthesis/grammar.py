""" This module defines the abstract syntax tree structures for the output
language. The `Program` dataclass is supposed to be top-level node of the AST.
"""

from typing import (Any, List, Optional, Dict, Union, NoReturn)
from dataclasses import dataclass
from enum import Enum


@dataclass(unsafe_hash=True)
class POI:
    """ Point Of Insertion. By convention, we allow inserting new
    statements strictly at the end of the list of already existing ones. """
    stmts:List["Stmt"]
    expr:"Expr"
    def __init__(self, stmts=None, expr=None):
        self.stmts = stmts if stmts is not None else []
        self.expr = expr if expr is not None else NoneExpr()


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

@dataclass(frozen=True)
class VRefExpr:
    """ Expression - reference to a variable """
    vname: VName

@dataclass(frozen=True)
class ConstExpr:
    """ Expression - constant """
    val: Union[bool, int, float, complex]

trueExpr = ConstExpr(True)
falseExpr = ConstExpr(False)

@dataclass(frozen=True)
class NoneExpr:
    """ Alias for None """
    pass

class ControlFlowStyle(Enum):
    Python = 0
    Catalyst = 1
    JAX = 2

@dataclass
class CondExpr:
    """ Expression - conditional """
    cond: Expr
    trueBranch: POI
    falseBranch: Optional[POI]
    style: ControlFlowStyle

@dataclass
class ForLoopExpr:
    """ Expression - for loop """
    loopvar: VName
    lbound: Expr
    ubound: Expr
    body: POI
    style: ControlFlowStyle

@dataclass
class WhileLoopExpr:
    """ Expression - while loop """
    loopvar: VName
    cond: Expr
    body: POI
    style: ControlFlowStyle

@dataclass(frozen=True)
class FCallExpr:
    """ Expression - calling a callable """
    expr: Union[FName, CondExpr, ForLoopExpr, WhileLoopExpr]
    args: List[Expr]

Stmt = Union["AssignStmt", "FDefStmt", "RetStmt"]

@dataclass
class AssignStmt:
    """ Statement - variable assignemnt or a function call """
    vname: Optional[VName]
    expr: Expr

@dataclass
class FDefStmt:
    """ Statement - function declaration """
    fname: FName
    args: List[VName]
    body: POI
    qwires: Optional[int] = None
    qdevice: Optional[str] = None

@dataclass
class RetStmt:
    """ Statement - return """
    expr: Optional[Expr]

Program = FDefStmt
""" Top-level program is a function declaration """


def assert_never(x: Any) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))


