import sys
from typing import Any, Dict, Tuple, Union, Callable, Set, List
from hypothesis import given, note, settings
from inspect import signature as python_signature, _empty as inspect_empty

import jax.numpy as jnp
from pytest import mark
from dataclasses import astuple

from catalyst.synthesis.grammar import (Expr, RetStmt, FCallExpr, VName, FName, VRefExpr, signature
                                        as expr_signature, isinstance_expr, innerdefs1, AssignStmt)
from catalyst.synthesis.pprint import pstr_builder, pstr_stmt, pstr_expr, pprint
from catalyst.synthesis.builder import build
from catalyst.synthesis.exec import compilePOI, evalPOI
from catalyst.synthesis.generator import greedy
from catalyst.synthesis.hypothesis import *


VERBOSE:bool = True

ExprPart = Callable[[POI],Union[Expr,"ExprPart"]]

def compilePOI_(*args, **kwargs):
    o,code = compilePOI(*args, **kwargs)
    note("Generated Python code is:")
    note(code)
    return o,code

def evalPOI_(p:POI, use_qjit=True, **kwargs):
    o,code = compilePOI_(p, use_qjit=use_qjit, **kwargs)
    return evalPOI(o)


def saturate(e:Union[Expr, ExprPart], val:POI) -> Expr:
    return saturate(e(val), val) if isinstance(e,Callable) else e


def mkCallExpr1(x:Expr, arg:Expr) -> Expr:
    """ Produces `FCallExpr(x, [arg]*N)` """
    s = expr_signature(x)
    assert s is not None, f"{x} is not a callable Expr"
    fname, anames = astuple(s)
    return FCallExpr(x, [arg for _ in anames])


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_cflow(x:callable):
    pprint(saturate(x, POI.fE(ConstExpr(33))))


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_fdef_ctflow(x:callable):
    pprint(FDefStmt(FName("main"), [], POI.fromExpr(saturate(x, POI.fE(ConstExpr(33))))))


@given(x=one_of([whileloops(), forloops(), conds()]))
def test_pprint_ret_ctflow(x:callable):
    pprint(RetStmt(saturate(x,POI.fE(ConstExpr(33)))))


@given(x=one_of([conds(),whileloops(),forloops()]))
def test_eq_expr(x):
    xa=saturate(x, POI())
    xb=saturate(x, POI())
    assert xa is not xb
    assert xa == xb
    xc=saturate(x, POI.fE(saturate(x,POI())))
    assert xa != xc


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False))
@settings(max_examples=10)
def test_eval_const(x, use_qjit):
    assert jnp.array([x]) == evalPOI_(POI.fE(ConstExpr(jnp.array([x]))), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False), c=conds())
@settings(max_examples=10)
def test_eval_cond(c, x, use_qjit):
    jx = jnp.array([x])
    x2 = FCallExpr(c(POI.fE(ConstExpr(jx)))(POI.fE(ConstExpr(jx))),[])
    assert jx == evalPOI_(POI.fE(x2), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=forloops(lvars=just(VName('i')),svars=just(VName('s'))))
@settings(max_examples=10)
def test_eval_for(l, x, use_qjit):
    jx = jnp.array([x])
    r = FCallExpr(l(POI.fE(VRefExpr(VName('s')))),[ConstExpr(jx)])
    assert jx == evalPOI_(POI.fE(r), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=whileloops(lvars=just(VName('i')),
                    lexprs=just(falseExpr)))
@settings(max_examples=10)
def test_eval_while(l, x, use_qjit):
    jx = jnp.array([x])
    r = FCallExpr(l(POI.fE(VRefExpr(VName('i')))),[ConstExpr(jx)])
    assert jx == evalPOI_(POI.fE(r), use_qjit)


@given(g=qgates, m=qmeasurements)
@settings(max_examples=10)
def test_eval_qops(g,m):
    evalPOI_(POI([AssignStmt.fE(g),RetStmt(m)]), use_qjit=True,
             qdevice="lightning.qubit", qwires=1)


def test_build_mutable_layout():
    l = WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst)
    c = CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst)

    l_poi = l.body
    c_poi1 = c.trueBranch
    c_poi2 = c.falseBranch

    b_poi = POI()
    b = build(b_poi)
    assert len(b.pois)==1
    assert b.pois[0].poi is b_poi

    poi1 = POI.fromExpr(l)
    b.update(0, poi1)
    assert len(b.pois)==2
    assert b.pois[0].poi is b_poi
    assert b.pois[1].poi is l_poi

    poi2 = POI.fromExpr(c)
    b.update(1, poi2)
    # pprint(b)
    assert len(b.pois)==4
    assert b.pois[0].poi is b_poi
    assert b.pois[1].poi is l_poi
    assert b.pois[2].poi is c_poi1
    assert b.pois[3].poi is c_poi2
    # pprint(b)
    b.update(0, POI())
    assert len(b.pois)==1


def test_build_destructive_update():
    l = WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst)
    c = CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst)
    b = build(POI())
    b.update(0, POI.fromExpr(l))
    b.update(1, POI.fromExpr(c))
    assert len(b.pois)==4
    s1 = pstr_builder(b)
    b.update(0, b.pois[0].poi)
    assert len(b.pois)==4
    s2 = pstr_builder(b)
    assert s1 == s2



# def test_build_context():
#     l = WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst)
#     b = build(POI())

# def run_build_controlflow(o, i):
#     d = ConstExpr(0)
#     i2 = bind1(part1(i), lambda vs: (lambda _ : vs[0] if vs else d))
#     o2 = bind1(part1(o), lambda vs: (lambda poi: closeargs(i2(poi), vs[0]) if vs else d))
#     pprint(o2(POI()))

#     # pprint(build(RetStmt(part1(o)(POI()))).insert_statement(0,RetStmt(part1(i)(POI()))))
#     # pprint(build(RetStmt(part1(o)(POI()))).insert_statement(0,RetStmt(ConstExpr(33))))
#     # pprint(build(POI.fromExpr(part1(o)(POI()))).append_expr(0, lambda _ : ConstExpr(33)))
#     b = build(
#         POI()
#     ).append_expr(
#         0, lambda ctx, var: closeargs(part1(o)(POI()),ConstExpr(0))
#     ).append_expr(
#         1, lambda ctx, var: closeargs(part1(i)(POI()),VRefExpr(ctx.vscope[-1]))
#     ).append_expr(
#         2, lambda ctx, var: VRefExpr(ctx.vscope[-1])
#     )
#     # part1(i)(POI.fromExpr(VRefExpr(ctx.vscope[-1]))
#     # print(b)
#     pprint(b)
#     # pprint(build(POI.fromExpr(part1(o)(POI()))).append_expr(0, lambda var : part1(i)(POI.fromExpr(var))))

# @given(o=one_of([whileloops(), forloops(), conds()]),
#        i=one_of([whileloops(), forloops(), conds()]))
# @settings(max_examples=1)
# def test_build_controlflow(o, i):
#     d = ConstExpr(0)
#     i2 = bind1(part1(i), lambda var: var[0] if var else d)
#     o2 = bind1(part1(o), lambda var: (lambda poi: closeargs(i2(poi), var[0]) if var else d))
#     pprint(o2(POI()))
#     # assert False



sample_spec:Dict[Expr,int] = {
    WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst) : 1,
    ForLoopExpr(VName("i"), ConstExpr(0), ConstExpr(10), POI(), ControlFlowStyle.Catalyst) : 2,
    # CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst) : 1,
}

def run_greedy():
    for b in greedy(sample_spec):
        pprint(b)


