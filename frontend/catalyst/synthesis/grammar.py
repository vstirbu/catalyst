""" Output program grammar definitons """

from typing import (Any, List, Optional, Dict, Union, NoReturn)
from dataclasses import dataclass

@dataclass(frozen=True)
class VName:
    """ Alias for strings representing variable names """
    val: str

@dataclass(frozen=True)
class FName:
    """ Alias for strings representing function names """
    val: str

Expr = Union["VRefExpr","FCallExpr", "ConstExpr"]

@dataclass
class FCallExpr:
    """ Expression - function call """
    fname: FName
    args: List[Expr]

@dataclass
class VRefExpr:
    """ Expression - reference to a variable """
    vname: VName

@dataclass
class ConstExpr:
    """ Expression - constant """
    val: Union[bool, int, float, complex]

TrueExpr = ConstExpr(True)
FalseExpr = ConstExpr(False)

@dataclass
class POIExpr:
    """ Expression - Point Of Insertion """
    pass

Stmt = Union["AssignStmt", "CondStmt", "WhileLoopStmt", "FDefStmt", "RetStmt"]

@dataclass
class POIStmt:
    """ Statement - Point Of Insertion """
    stmts:List[Stmt]
    def __init__(self, stmts=None):
        self.stmts = stmts if stmts is not None else []

@dataclass
class AssignStmt:
    """ Statement - variable assignemnt or a function call """
    vname: Optional[VName]
    expr: Expr

@dataclass
class CondStmt:
    """ Statement - conditional """
    cond: Expr
    trueBranch: POIStmt
    falseBranch: Optional[POIStmt]

@dataclass
class WhileLoopStmt:
    """ Statement - while loop """
    cond: Expr
    body: POIStmt

@dataclass
class FDefStmt:
    """ Statement - function declaration """
    decorators: List[FName]
    fname: FName
    args: List[VName]
    body: POIStmt

@dataclass
class RetStmt:
    """ Statement - return """
    expr: Optional[Expr]

Program = FDefStmt
""" Top-level program is a function declaration """


def assert_never(x: Any) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))


