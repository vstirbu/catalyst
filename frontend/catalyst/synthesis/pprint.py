""" Output program pretty-printer functions

TODO: Decide on https://peps.python.org/pep-0634/
"""

from typing import Union, List, Optional, NoReturn

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondStmt, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POIStmt, assert_never)

def pprint_expr(e:Expr) -> str:
    if False:
        pass
    elif isinstance(e, FCallExpr):
        return f"{e.fname.val}({','.join([pprint_expr(a) for a in e.args])})"
    elif isinstance(e, VRefExpr):
        return e.vname.val
    elif isinstance(e, ConstExpr):
        if isinstance(e.val, int):
            return f"int({e.val})"
        elif isinstance(e.val, bool):
            return f"bool({e.val})"
        elif isinstance(e.val, float):
            return f"float({e.val})"
        elif isinstance(e.val, complex):
            return f"complex({e.val})"
        else:
            assert_never(e.val)
    else:
        assert_never(e)

TABSTOP:int = 4

def pprint_stmt(s:Stmt, indent:int=0) -> List[str]:
    def _p(lines):
        return [' '*(indent*TABSTOP) + line for line in lines]
    def _ne(l:list)->list:
        return sum(l,[]) if len(l)>0 else [' '*TABSTOP + "pass"]
    if False:
        pass
    elif isinstance(s, AssignStmt):
        if s.vname is not None:
            return _p([f"{s.vname.val} = {pprint_expr(s.expr)}"])
        else:
            return _p(["{pprint_expr(s.expr)}"])
    elif isinstance(s, CondStmt):
        true_part = _p([f"if {pprint_expr(s.cond)} then:"] +
                       _ne([pprint_stmt(s, 1) for s in s.trueBranch.stmts]))
        false_part = _p(["else:"] +
                       _ne([pprint_stmt(s, 1) for s in s.falseBranch.stmts])) \
                     if s.falseBranch else []
        return true_part + false_part
    elif isinstance(s, WhileLoopStmt):
        return _p([f"while {pprint_expr(s.cond)}:"] +
                  _ne([pprint_stmt(s, 1) for s in s.body.stmts]))
    elif isinstance(s, FDefStmt):
        return _p([f"@{d.val}" for d in s.decorators] +
                  [f"def {s.fname.val}({','.join([a.val for a in s.args])}):"] +
                  _ne([pprint_stmt(s, 1) for s in s.body.stmts]))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            return _p([f"return {pprint_expr(s.expr)}"] )
        else:
            return _p(["return"])
    else:
        assert_never(s)


def pprint_prog(p:Program, indent:int=0) -> List[str]:
    """ Pretty-print the program """
    return pprint_stmt(p, indent)

def pprint(p:Union[Program, Stmt, Expr]) -> None:
    """ Prints the program on the console
    FIXME: Find out how not to repeat Stmt and Expr definitions
    """
    if isinstance(p, Program):
        print('\n'.join(pprint_prog(p)))
    elif isinstance(p, (AssignStmt, CondStmt, WhileLoopStmt, FDefStmt, RetStmt)):
        print('\n'.join(pprint_stmt(p)))
    elif isinstance(p, (VRefExpr, FCallExpr, ConstExpr)):
        print(pprint_expr(p))
    else:
        assert_never(p)


