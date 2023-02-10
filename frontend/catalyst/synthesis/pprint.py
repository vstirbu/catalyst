""" Output program pretty-printer functions

TODO: Decide on https://peps.python.org/pep-0634/
"""

from typing import Union, List, Optional, NoReturn, Callable

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondStmt, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POIStmt, ForLoopStmt, ControlFlowStyle,
                      assert_never)

from .generator import (POITracker)

DEFAULT_QDEVICE = "catalyst-lightning"

def pprint_expr(e:Expr) -> str:
    if isinstance(e, FCallExpr):
        return f"{e.fname.val}({', '.join([pprint_expr(a) for a in e.args])})"
    elif isinstance(e, VRefExpr):
        return e.vname.val
    elif isinstance(e, ConstExpr):
        if isinstance(e.val, bool): # Should be above 'int'
            return f"bool({e.val})"
        elif isinstance(e.val, int):
            return f"int({e.val})"
        elif isinstance(e.val, float):
            return f"float({e.val})"
        elif isinstance(e.val, complex):
            return f"complex({e.val})"
        else:
            assert_never(e.val)
    else:
        assert_never(e)

TABSTOP:int = 4

HintPrinter = Callable[[POIStmt],List[str]]

def pprint_stmt(s:Stmt, indent:int=0,
                hint:Optional[HintPrinter]=None) -> List[str]:
    def _p(lines):
        return [' '*(indent*TABSTOP) + line for line in lines]
    def _ne(l:list)->list:
        return sum(l,[]) if len(l)>0 else [' '*TABSTOP + "pass"]
    def _hl(poi):
        h = hint if hint else (lambda _:[])
        return [' '*TABSTOP + "# " + h for h in [f"poi {id(poi)}"]+h(poi) if h]

    if False:
        pass
    elif isinstance(s, AssignStmt):
        if s.vname is not None:
            return _p([f"{s.vname.val} = {pprint_expr(s.expr)}"])
        else:
            return _p(["{pprint_expr(s.expr)}"])
    elif isinstance(s, CondStmt):
        if s.style == ControlFlowStyle.Python:
            true_part = _p([f"if {pprint_expr(s.cond)} then:"] +
                           _ne([pprint_stmt(s, 1, hint) for s in s.trueBranch.stmts]) +
                           _hl(s.trueBranch)
                           )
            false_part = _p(["else:"] +
                           _ne([pprint_stmt(s, 1, hint) for s in s.falseBranch.stmts]) +
                           _hl(s.falseBranch)) \
                         if s.falseBranch else []
            return true_part + false_part
        else:
            assert_never(s.style)
    elif isinstance(s, ForLoopStmt):
        if s.style == ControlFlowStyle.Python:
            return _p([f"for {s.loopvar} in range({pprint_expr(s.lbound)}, "
                                                f"{pprint_expr(s.ubound)}):"] +
                      _ne([pprint_stmt(s, 1, hint) for s in s.body.stmts]) +
                      _hl(s.body))
        else:
            assert_never(s.style)
    elif isinstance(s, WhileLoopStmt):
        return _p([f"while {pprint_expr(s.cond)}:"] +
                  _ne([pprint_stmt(s, 1, hint) for s in s.body.stmts]) +
                  _hl(s.body))
    elif isinstance(s, FDefStmt):
        qdevice = s.qdevice if s.qdevice is not None else DEFAULT_QDEVICE
        qfunc = [f"@qml.qnode(qml.device(\"{qdevice}\", wires={s.qwires}))"] \
                if s.qwires is not None else []
        return _p(qfunc +
                  [f"def {s.fname.val}({', '.join([a.val for a in s.args])}):"] +
                  _ne([pprint_stmt(s, 1, hint) for s in s.body.stmts]) +
                  _hl(s.body))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            return _p([f"return {pprint_expr(s.expr)}"] )
        else:
            return _p(["return"])
    else:
        assert_never(s)

def pprint_tracker(t:POITracker, indent:int=0) -> List[str]:
    def _hp(poi:POIStmt) -> List[str]:
        for poic in t.pois:
            if poi is poic.poi:
                return [', '.join(v.val for v in poic.ctx.get_vscope())]
        return []
    return pprint_stmt(t.stmt, indent, _hp)

def pprint_prog(p:Program, indent:int=0) -> List[str]:
    """ Pretty-print the program """
    return pprint_stmt(p, indent)

def pprint(p:Union[POITracker, Program, Stmt, Expr]) -> None:
    """ Prints the program on the console
    FIXME: Find out how not to repeat Stmt and Expr definitions
    """
    if isinstance(p, POITracker):
        print('\n'.join(pprint_tracker(p)))
    elif isinstance(p, Program):
        print('\n'.join(pprint_prog(p)))
    elif isinstance(p, (AssignStmt, CondStmt, WhileLoopStmt, FDefStmt, RetStmt)):
        print('\n'.join(pprint_stmt(p)))
    elif isinstance(p, (VRefExpr, FCallExpr, ConstExpr)):
        print(pprint_expr(p))
    else:
        assert_never(p)


