import sys
import math
import cmath
from typing import Any, Dict, Tuple, Union
from hypothesis import given, note, settings
from inspect import signature as python_signature, _empty as inspect_empty

import jax.numpy as jnp
import pennylane as qml
from catalyst.synthesis.grammar import (Expr, RetStmt, FCallExpr, VName, FName, VRefExpr, signature
                                        as expr_signature, isinstance_expr)
from catalyst.synthesis.hypothesis import *
from catalyst.synthesis.pprint import pstr_stmt, pstr_expr, pprint
from catalyst.synthesis.builder import build
from catalyst import qjit, for_loop, while_loop, cond

from pytest import mark


VERBOSE:bool = True

def log(s:Union[list,str]) -> None:
    s='\n'.join(s) if isinstance(s, list) else s
    try:
        note(s)
    except TypeError:
        if VERBOSE:
            print(s, file=sys.stderr)

def mkExprKwargs(arg:Expr, x:callable) -> Dict[str,Expr]:
    """Makes `kwarg` dict passing `arg` to all `x` parameters which dont have default values """
    return {k:arg for k,v in dict(python_signature(x).parameters).items()
            if v.default==inspect_empty}


def mkCallExpr1(x:Expr, arg:VName) -> Expr:
    """ Produces `FCallExpr` passing `arg` to all arguments """
    s = expr_signature(x)
    assert s is not None, f"{x} is not a callable Expr"
    fname, anames = s
    return FCallExpr(x, [arg for _ in anames])


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_fdef(x:callable):
    e = x(**mkExprKwargs(POI(), x))
    main = FDefStmt(FName("main"), [], POI.fromExpr(mkCallExpr1(e, ConstExpr(0))))
    pprint(main)


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_fcall(x:callable):
    arg=VRefExpr(VName('var1'))
    pprint(RetStmt(mkCallExpr1(x(**mkExprKwargs(POI(),x)),arg)))


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_controlflow(x:callable):
    pprint(x(**mkExprKwargs(POI(), x)))


@given(x=one_of([conds(),whileloops(),forloops()]))
def test_eq_expr(x:callable):
    args=mkExprKwargs(POI(), x)
    assert x(**args)==x(**args)
    args2=mkExprKwargs(POI([],x(**args)), x)
    assert x(**args)!=x(**args2)


PythonCode = str
PythonObj = Any


def compileExpr(e :Expr, use_qjit:bool=True) -> Tuple[PythonObj, PythonCode]:
    main = FDefStmt(FName("main"), [], POI.fromExpr(
        e if expr_signature(e) is None else mkCallExpr1(e, CondExpr(0))), qjit=use_qjit)
    code = '\n'.join(pstr_stmt(main))
    log(["Compiled program is:", code])
    o = compile(code, "<compileExpr>", "single")
    return (o, code)


def evalExpr(e: Union[Expr,PythonObj], use_qjit:bool=True) -> Any:
    o = compileExpr(e, use_qjit)[0] if isinstance_expr(e) else e
    gctx, lctx = {}, {}
    gctx.update({'qml': qml,
                 'for_loop':for_loop,
                 'while_loop':while_loop,
                 'cond':cond,
                 'qjit':qjit,
                 'Array':jnp.array,
                 'int64':jnp.int64,
                 'complex128':jnp.complex128,
                 'inf':math.inf,
                 'nan':math.nan,
                 'infj':cmath.infj,
                 'nanj':cmath.nanj})
    s = exec(o, gctx, lctx)
    r = eval("main()", gctx, lctx)
    return r


@mark.parametrize('use_qjit',[True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False))
@settings(max_examples=10)
def test_eval_expr(x, use_qjit):
    assert jnp.array([x]) == evalExpr(ConstExpr(jnp.array([x])), use_qjit)


@given(x=one_of([whileloops(), forloops(), conds()]))
@settings(max_examples=1)
def test_build_controlflow(x:callable):
    kwargs1 = mkExprKwargs(POI(), x)
    kwargs2 = mkExprKwargs(POI.fromExpr(x(**kwargs1)), x)
    pprint(x(**kwargs2))
    b = build(RetStmt(x(**kwargs2)))
    pprint(b)
    assert False



