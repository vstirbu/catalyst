""" Output program pretty-printer functions

TODO: Decide on https://peps.python.org/pep-0634/
"""

from typing import Tuple, Union, List, Optional, NoReturn, Callable
from dataclasses import dataclass

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, NoneExpr, POI, ForLoopStmt, ControlFlowStyle,
                      assert_never)

from .builder import (Builder)

DEFAULT_QDEVICE = "catalyst-lightning"

@dataclass
class PStrState:
    indent:int = 0
    catalyst_cf_suffix:int = 0

    def tabulate(self) -> "PStrState":
        return PStrState(self.indent+1, self.catalyst_cf_suffix)
    def issue(self, name) -> Tuple["PStrState",str]:
        s2 = PStrState(self.indent, self.catalyst_cf_suffix+1)
        return s2, f"{name}{self.catalyst_cf_suffix}"

def _in(st, lines):
    """ Indent """
    return [' '*(st.indent*TABSTOP) + line for line in lines]

def _ne(st, l:list)->list:
    """ Check non-empty """
    return sum(l,[]) if len(l)>0 else [' '*((st.indent+1)*TABSTOP) + "pass"]

def _hi(st, hint, poi):
    """ Print a hint """
    hlines = hint(poi) if hint else []
    if hlines:
        hlines = [f"poi {id(poi)}"] + hlines
    return [' '*(st.indent+1)*TABSTOP + "# " + h for h in hlines]


TABSTOP:int = 4

HintPrinter = Callable[[POI],List[str]]

def pstr_expr(expr:Expr,
              state:Optional[PStrState]=None,
              hint:Optional[HintPrinter]=None) -> Tuple[List[str],str]:
    e = expr
    st = state if state else PStrState()
    if isinstance(e, FCallExpr):
        accs,acce = list(),list()
        for ea in e.args:
            lss,le = pstr_expr(ea, state, hint)
            accs.extend(lss)
            acce.append(le)
        return accs,f"{e.fname.val}({', '.join(acce)})"
    elif isinstance(e, CondExpr):
        if e.style == ControlFlowStyle.Python:
            assert_never(e.style)
        elif e.style == ControlFlowStyle.Catalyst:
            acc, lcond = pstr_expr(e.cond, st, hint)
            st1, nmcond = st.tabulate().issue("cond")
            true_part = (
                _in(st, [f"@cond({lcond})",
                         f"def {nmcond}():"]) +
                _ne(st, [pstr_stmt(s, st1, hint) for s in e.trueBranch.stmts]) +
                _hi(st, hint, e.trueBranch))
            false_part = (
                _in(st, [f"@{nmcond}.otherwise",
                         f"def {nmcond}():"]) +
                _ne(st, [pstr_stmt(s, st1, hint) for s in e.falseBranch.stmts]) +
                _hi(st, hint, e.falseBranch)) if e.falseBranch else []
            return acc + true_part + false_part, f"{nmcond}()"
        else:
            assert_never(e.style)
    elif isinstance(e, NoneExpr):
        return [],"None"
    elif isinstance(e, VRefExpr):
        return [],e.vname.val
    elif isinstance(e, ConstExpr):
        if isinstance(e.val, bool): # Should be above 'int'
            return [],f"bool({e.val})"
        elif isinstance(e.val, int):
            return [],f"int({e.val})"
        elif isinstance(e.val, float):
            return [],f"float({e.val})"
        elif isinstance(e.val, complex):
            return [],f"complex({e.val})"
        else:
            assert_never(e.val)
    else:
        assert_never(e)

def pstr_stmt(s:Stmt,
              state:Optional[PStrState]=None,
              hint:Optional[HintPrinter]=None) -> List[str]:
    st:PStrState = state if state is not None else PStrState()
    indent = st.indent

    if False:
        pass
    elif isinstance(s, AssignStmt):
        acc, lexpr = pstr_expr(s.expr, state, hint)
        if s.vname is not None:
            return acc + _in(st, [f"{s.vname.val} = {lexpr}"])
        else:
            return acc + _in(st, [lexpr])
    elif isinstance(s, ForLoopStmt):
        accL, lexprL = pstr_expr(s.lbound, state, hint)
        accU, lexprU = pstr_expr(s.ubound, state, hint)
        if s.style == ControlFlowStyle.Python:
            st1 = st.tabulate()
            return (
                accL + accU +
                _in(st, [f"for {s.loopvar.val} in range({lexprL}, "
                                                      f"{lexprU}):"]) +
                _ne(st, [pstr_stmt(s, st1, hint) for s in s.body.stmts]) +
                _hi(st, hint, s.body))
        else:
            assert_never(s.style)
    elif isinstance(s, WhileLoopStmt):
        acc, lexpr = pstr_expr(s.cond, state, hint)
        if s.style == ControlFlowStyle.Python:
            st1 = st.tabulate()
            return (
                acc +
                _in(st, [f"while {lexpr}:"]) +
                _ne(st, [pstr_stmt(s, st1, hint) for s in s.body.stmts]) +
                _hi(st, hint, s.body))
        else:
            assert_never(s.style)
    elif isinstance(s, FDefStmt):
        st1 = st.tabulate()
        qdevice = s.qdevice if s.qdevice is not None else DEFAULT_QDEVICE
        qfunc = [f"@qml.qnode(qml.device(\"{qdevice}\", wires={s.qwires}))"] \
                if s.qwires is not None else []
        return (
            _in(st, qfunc +
                [f"def {s.fname.val}({', '.join([a.val for a in s.args])}):"]) +
            _ne(st, [pstr_stmt(s, st1, hint) for s in s.body.stmts]) +
            _hi(st, hint, s.body))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            acc, lexpr = pstr_expr(s.expr, state, hint)
            return acc + _in(st, [f"return {lexpr}"] )
        else:
            return _in(st, ["return"])
    else:
        assert_never(s)

def pstr_tracker(t:Builder, state:Optional[PStrState]=None) -> List[str]:
    st = state if state is not None else PStrState()
    def _hp(poi:POI) -> List[str]:
        for poic in t.pois:
            if poi is poic.poi:
                return [', '.join(v.val for v in sorted(poic.ctx.get_vscope()))]
        return []
    return pstr_stmt(t.stmt, st, _hp)

def pstr_prog(p:Program, state:Optional[PStrState]=None) -> List[str]:
    """ Pretty-print the program """
    return pstr_stmt(p, state)

def pprint(p:Union[Builder, Program, Stmt, Expr]) -> None:
    """ Prints the program on the console
    FIXME: Find out how not to repeat Stmt and Expr definitions
    """
    if isinstance(p, Builder):
        print('\n'.join(pstr_tracker(p)))
    elif isinstance(p, Program):
        print('\n'.join(pstr_prog(p)))
    elif isinstance(p, (AssignStmt, WhileLoopStmt, FDefStmt, RetStmt,
                        ForLoopStmt)):
        print('\n'.join(pstr_stmt(p)))
    elif isinstance(p, (VRefExpr, FCallExpr, ConstExpr, CondExpr)):
        print(pstr_expr(p))
    else:
        assert_never(p)


