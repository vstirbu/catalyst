""" Output program pretty-printer functions

TODO: Decide on https://peps.python.org/pep-0634/
"""

from typing import Tuple, Union, List, Optional, NoReturn, Callable
from itertools import chain

from dataclasses import dataclass

try:
    from jax import Array as JaxArray
except ImportError:
    class JaxArray:
        def __str__(self):
            return "FakeJaxArray()"

from pennylane.numpy import tensor as PnpArray

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopExpr, FDefStmt, Program, RetStmt,
                      ConstExpr, NoneExpr, POI, ForLoopExpr, ControlFlowStyle,
                      assert_never, isinstance_expr, isinstance_stmt, ExprLike, bless_expr,
                      isinstance_exprlike)

from .builder import (Builder)

DEFAULT_QDEVICE = "lightning.qubit"

DEFAULT_CFSTYLE = ControlFlowStyle.Catalyst

HintPrinter = Callable[[POI],List[str]]

CFS = ControlFlowStyle

@dataclass
class PStrOptions:
    default_cfstyle:ControlFlowStyle = DEFAULT_CFSTYLE
    hint:Optional[HintPrinter] = None


@dataclass
class Suffix:
    """ Mutable counter helping to produce `unique` entity names. """
    val:int

@dataclass
class PStrState:
    """ The state to carry around the pretty-printing procedures. """
    indent:int = 0
    catalyst_cf_suffix:int = Suffix(0)

    def tabulate(self) -> "PStrState":
        return PStrState(self.indent+1, self.catalyst_cf_suffix)
        # self.indent+=1
        # return self
    def issue(self, name) -> Tuple["PStrState",str]:
        self.catalyst_cf_suffix.val+=1
        s2 = PStrState(self.indent, self.catalyst_cf_suffix)
        return s2, f"{name}{self.catalyst_cf_suffix.val-1}"
        # self.catalyst_cf_suffix+=1
        # return self, f"{name}{self.catalyst_cf_suffix-1}"

def _in(st:PStrState, lines:List[str]) -> List[str]:
    """ Indent `lines` according to the current settings """
    return [' '*(st.indent*TABSTOP) + line for line in lines]

def _ne(st:PStrState, ls:List[str]) -> List[str]:
    """ Issue `pass` (a Python keyword) as a substitute of empty stmt list. """
    return ls if len(ls)>0 else [' '*((st.indent+1)*TABSTOP) + "pass"]

def _hi(st:PStrState, opt:Optional[PStrOptions], poi:POI) -> List[str]:
    """ Issue a hint formatted as a Python comment """
    hlines = opt.hint(poi) if opt and opt.hint else []
    if len(hlines) > 0:
        hlines = [f"poi {id(poi)}"] + hlines
    return [' '*st.indent*TABSTOP + "# " + h for h in hlines]


TABSTOP:int = 4

def _style(s, opt):
    s = (opt.default_cfstyle if opt else DEFAULT_CFSTYLE) if s==ControlFlowStyle.Default else s
    assert s != ControlFlowStyle.Default, f"Concrete CF style is expected at this point"
    return s

def _isinfix(e:FCallExpr) -> bool:
    return isinstance(e.expr, VRefExpr) and all(c in "!>=<=+-/*" for c in e.expr.vname.val) and len(e.args)==2

def _parens(e:Expr, expr_str:str, opt) -> str:
    if isinstance(e, (VRefExpr, ConstExpr)) or (isinstance(e, FCallExpr) and not _isinfix(e)):
        return expr_str
    else:
        return f"({expr_str})"

def pstr_expr(expr:ExprLike,
              state:Optional[PStrState]=None,
              opt:Optional[PStrOptions]=None,
              arg_expr:Optional[List[Expr]]=None) -> Tuple[List[str],str]:
    e = bless_expr(expr)
    st = state if state else PStrState(0,Suffix(0))
    if isinstance(e, FCallExpr):
        return pstr_expr(e.expr, state, opt, arg_expr=e.args)
    elif isinstance(e, CondExpr):
        assert arg_expr is not None
        if _style(e.style, opt) == ControlFlowStyle.Python:
            acc, scond = pstr_expr(e.cond, st, opt)
            st1, svar = st.tabulate().issue("_cond")
            true_part = (
                _in(st, [f"if {scond}:"]) +
                _ne(st,
                    sum([pstr_stmt(s, st1, opt) for s in e.trueBranch.stmts], []) +
                    (pstr_stmt(AssignStmt(VName(svar), e.trueBranch.expr), st1, opt) if
                     e.trueBranch.expr else [])) +
                _hi(st1, opt, e.trueBranch))
            false_part = (
                _in(st, ["else:"]) +
                _ne(st,
                    sum([pstr_stmt(s, st1, opt) for s in e.falseBranch.stmts], []) +
                    (pstr_stmt(AssignStmt(VName(svar), e.falseBranch.expr), st1, opt) if
                     e.falseBranch.expr else [])) +
                _hi(st1, opt, e.falseBranch)) if e.falseBranch else []
            return (acc + true_part + false_part, svar)
        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            acc, lcond = pstr_expr(e.cond, st, opt)
            st1, nmcond = st.tabulate().issue("cond")
            true_part = (
                _in(st, [f"@cond({lcond})",
                         f"def {nmcond}():"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.trueBranch.stmts], []) +
                        (pstr_stmt(RetStmt(e.trueBranch.expr), st1, opt) if e.trueBranch.expr else [])) +
                _hi(st1, opt, e.trueBranch))
            false_part = (
                _in(st, [f"@{nmcond}.otherwise",
                         f"def {nmcond}():"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.falseBranch.stmts], []) +
                        (pstr_stmt(RetStmt(e.falseBranch.expr), st1, opt) if e.falseBranch.expr else [])) +
                _hi(st1, opt, e.falseBranch)) if e.falseBranch else []
            return (acc + true_part + false_part, f"{nmcond}()")
        else:
            assert_never(e.style)
    elif isinstance(e, ForLoopExpr):
        assert len(arg_expr)==1
        if _style(e.style, opt) == ControlFlowStyle.Python:
            st1 = st.tabulate()
            accArg = pstr_stmt(AssignStmt(e.statevar, arg_expr[0]), st, opt) if e.statevar else []
            accL, lexprL = pstr_poi(e.lbound, st, opt)
            accU, lexprU = pstr_poi(e.ubound, st, opt)
            return (
                accArg + accL + accU +
                _in(st, [f"for {e.loopvar.val} in range({lexprL},{lexprU}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(AssignStmt(e.statevar,e.body.expr), st1, opt)
                         if e.body.expr and e.statevar else [])) +
                _hi(st1, opt, e.body), (e.statevar.val if e.statevar else "None"))

        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            accArg, sarg = pstr_expr(arg_expr[0], st, opt)
            accL, lexprL = pstr_poi(e.lbound, st, opt)
            accU, lexprU = pstr_poi(e.ubound, st, opt)
            st1, nforloop = st.tabulate().issue("forloop")
            args = ','.join([e.loopvar.val] + ([e.statevar.val] if e.statevar else []))
            accLoop = (
                _in(st, [f"@for_loop({lexprL},{lexprU},1)",
                         f"def {nforloop}({args}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(RetStmt(e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body))
            return (accArg + accL + accU + accLoop, f"{nforloop}({sarg})")
        else:
            assert_never(e.style)
    elif isinstance(e, WhileLoopExpr):
        assert len(arg_expr)==1
        if _style(e.style, opt) == ControlFlowStyle.Python:
            accArg = pstr_stmt(AssignStmt(e.statevar, arg_expr[0]), st, opt)
            accCond, lexpr = pstr_expr(e.cond, st, opt)
            st1, svar = st.tabulate().issue("_whileloop")
            return (
                accArg +
                accCond +
                _in(st, [f"while {lexpr}:"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(AssignStmt(e.statevar, e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body),
                e.statevar.val)
        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            accArg, sarg = pstr_expr(arg_expr[0], st, opt)
            accCond, lexpr = pstr_expr(e.cond, st, opt)
            st1, nwhileloop = st.tabulate().issue("whileloop")
            return (
                accArg + accCond +
                _in(st, [f"@while_loop(lambda {e.statevar.val}:{lexpr})",
                         f"def {nwhileloop}({e.statevar.val}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts],[]) +
                        (pstr_stmt(RetStmt(e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body), f"{nwhileloop}({sarg})")
        else:
            assert_never(s.style)
    elif isinstance(e, NoneExpr):
        return [],"None"
    elif isinstance(e, VRefExpr):
        if arg_expr is None:
            return [],e.vname.val
        else:
            acc_body,args = list(),list()
            for ea in arg_expr:
                lss,le = pstr_expr(ea, state, opt)
                acc_body.extend(lss)
                args.append(le)
            return acc_body, (
                f"{_parens(arg_expr[0], args[0], opt)} {e.vname.val} {_parens(arg_expr[1], args[1], opt)}"
                if len(arg_expr)==2 and _isinfix(FCallExpr(e, arg_expr)) else
                # if (all(c in "!>=<=+-/*" for c in e.vname.val) and len(arg_expr)==2) else
                f"{e.vname.val}({', '.join(args)})" )
    elif isinstance(e, ConstExpr):
        if isinstance(e.val, bool): # Should be above 'int'
            return [],f"{e.val}"
        elif isinstance(e.val, int):
            return [],f"{e.val}"
        elif isinstance(e.val, float):
            return [],f"{e.val}"
        elif isinstance(e.val, complex):
            return [],f"{e.val}"
        elif isinstance(e.val, JaxArray):
            return [],f"Array({e.val.tolist()},dtype={str(e.val.dtype)})"
        elif isinstance(e.val, PnpArray):
            return [],f"Array({e.val.tolist()},dtype={str(e.val.dtype)})"
        else:
            assert_never(e.val)
    else:
        assert_never(e)

def pstr_stmt(s:Stmt,
              state:Optional[PStrState]=None,
              opt:Optional[PStrOptions]=None) -> List[str]:
    st:PStrState = state if state is not None else PStrState(0,Suffix(0))
    if isinstance(s, AssignStmt):
        acc, lexpr = pstr_expr(s.expr, st, opt)
        return acc + _in(st, [f"{s.vname.val if s.vname else '_'} = {lexpr}"])
    elif isinstance(s, FDefStmt):
        st1 = st.tabulate()
        qjit = ["@qjit"] if s.qjit else []
        qfunc = [f"@qml.qnode(qml.device(\"{s.qnode_device or 'default.qubit'}\", wires={s.qnode_wires or 1}))"] \
                if (s.qnode_device is not None) or (s.qnode_wires is not None) else []
        return (
            _in(st, qjit + qfunc +
                [f"def {s.fname.val}({', '.join([a.val for a in s.args])}):"]) +
            _ne(st, sum([pstr_stmt(s, st1, opt) for s in s.body.stmts],[]) +
                    (pstr_stmt(RetStmt(s.body.expr), st1, opt) if s.body.expr else [])) +
            _hi(st1, opt, s.body))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            acc, lexpr = pstr_expr(s.expr, st, opt)
            return acc + _in(st, [f"return {lexpr}"] )
        else:
            return _in(st, ["return"])
    else:
        assert_never(s)

def pstr_poi(p:POI, state=None, opt=None, arg_expr=None) -> Tuple[List[str],str]:
    st = state if state is not None else PStrState(0,Suffix(0))
    lines, e = pstr_expr(p.expr, st, opt, arg_expr)
    return (sum((pstr_stmt(s, st, opt) for s in p.stmts), []) +
            lines + _hi(st, opt, p), e)

def builder_hint_printer(b):
    def _hp(poi:POI) -> List[str]:
        for poic in b.pois:
            if poi is poic.poi:
                return [', '.join(v.val for v in sorted(poic.ctx.get_vscope()))]
        return []
    return _hp

def pstr_builder(b:Builder, st=None, opt=None) -> Tuple[List[str],str]:
    opt = opt if opt else PStrOptions(DEFAULT_CFSTYLE, builder_hint_printer(b))
    return pstr_poi(b.pois[0].poi, st, opt, arg_expr=[VRefExpr(VName('<?>'))])

def pstr_prog(p:Program, state=None, opt=None) -> List[str]:
    """ Pretty-print the program """
    return pstr_stmt(p, state, opt)


def pstr(p:Union[Builder, Program, Stmt, ExprLike], default_cfstyle=DEFAULT_CFSTYLE) -> str:
    """ Prints the program on the console
    FIXME: Find out how not to repeat Stmt and Expr definitions
    """
    if isinstance(p, Builder):
        opt = PStrOptions(default_cfstyle, builder_hint_printer(p))
        lines, tail = pstr_builder(p, None, opt)
        return '\n'.join(lines+[f"## {tail} ##"])
    else:
        opt = PStrOptions(default_cfstyle)
        if isinstance(p, Program):
            return '\n'.join(pstr_prog(p, None, opt))
        elif isinstance_stmt(p):
            return '\n'.join(pstr_stmt(p, None, opt))
        elif isinstance_exprlike(p):
            stmts,tail = pstr_expr(p, None, opt, arg_expr=[VRefExpr(VName("<?>"))])
            return '\n'.join(stmts + [f"## {tail} ##"])
        else:
            assert_never(p)


def pprint(p:Union[Builder, Program, Stmt, ExprLike], default_cfstyle=CFS.Catalyst) -> None:
    print(pstr(p, default_cfstyle=default_cfstyle))

