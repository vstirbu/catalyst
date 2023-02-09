""" This module defines the abstract syntax tree structures for the output
language. The `Program` dataclass is supposed to be top-level node of the AST.
"""

from typing import (Any, List, Optional, Dict, Union, NoReturn)
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class VName:
    """ Alias for strings representing variable names """
    val: str

@dataclass(frozen=True)
class FName:
    """ Alias for strings representing function names """
    val: str

Expr = Union["VRefExpr","FCallExpr", "ConstExpr"]

@dataclass(frozen=True)
class FCallExpr:
    """ Expression - a function call """
    fname: FName
    args: List[Expr]

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

Stmt = Union["AssignStmt", "CondStmt", "WhileLoopStmt", "FDefStmt", "RetStmt"]

@dataclass(unsafe_hash=True)
class POIStmt:
    """ Statement - Point Of Insertion. By convention, we allow inserting new
    statements strictly at the end of the list of already existing ones. """
    stmts:List[Stmt]
    def __init__(self, stmts=None):
        self.stmts = stmts if stmts is not None else []

@dataclass
class AssignStmt:
    """ Statement - variable assignemnt or a function call """
    vname: Optional[VName]
    expr: Expr


class ControlFlowStyle(Enum):
    Python = 0
    Catalyst = 1
    JAX = 2

@dataclass
class CondStmt:
    """ Statement - conditional """
    cond: Expr
    trueBranch: POIStmt
    falseBranch: Optional[POIStmt]
    style: ControlFlowStyle

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


