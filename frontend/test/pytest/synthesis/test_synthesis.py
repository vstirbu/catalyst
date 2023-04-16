import os
import sys
from typing import Any, Dict, Tuple, Union, Callable, Set, List
from hypothesis import given, note, settings, Verbosity, assume
from inspect import signature as python_signature, _empty as inspect_empty
from tempfile import mktemp
from itertools import cycle

import jax.numpy as jnp
from pytest import mark
from dataclasses import astuple
from numpy.testing import assert_allclose

from catalyst.synthesis.grammar import (Expr, RetStmt, FCallExpr, VName, FName, VRefExpr, signature
                                        as expr_signature, isinstance_expr, innerdefs1, AssignStmt,
                                        lessExpr, addExpr, eqExpr, ControlFlowStyle as CFS, signature,
                                        Signature, bind, saturate_expr, saturates_expr1,
                                        saturates_poi1, saturate_expr1, saturate_poi1)

from catalyst.synthesis.pprint import pstr_builder, pstr_stmt, pstr_expr, pprint, pstr
from catalyst.synthesis.builder import build
from catalyst.synthesis.exec import compilePOI, evalPOI, runPOI, wrapInMain
from catalyst.synthesis.generator import control_flows
from catalyst.synthesis.hypothesis import *


def compilePOI_(*args, **kwargs):
    s = wrapInMain(*args, **kwargs)
    print("Generated Python statement is:")
    print('\n'.join(pstr_stmt(s)))
    return compilePOI(s)


def evalPOI_(p:POI, use_qjit=True, args:Optional[List[Tuple[Expr,Any]]]=None, **kwargs):
    arg_exprs = list(zip(*args))[0] if args is not None and len(args)>0 else []
    arg_all = args if args is not None else []
    o,code = compilePOI_(p, use_qjit=use_qjit, args=arg_exprs, **kwargs)
    return evalPOI(o, args=arg_all)


@given(d=data(), c=one_of([floats(), integers()])) # complexes() doesn't work
@settings(max_examples=10)
def test_eval_whiles(d, c):
    assume(c != 0)
    l = d.draw(whileloops(lexpr=lambda x: just(eqExpr(x,ConstExpr(c-c)))))
    def render(style):
        return POI.fE(saturates_expr1(ConstExpr(c-c),
                      saturates_poi1(ConstExpr(c),l(style=style))))

    r1 = evalPOI_(render(CFS.Python), use_qjit=False)
    r2 = evalPOI_(render(CFS.Catalyst), use_qjit=True)
    assert_allclose(r1, r2)


@given(x = one_of([forloops(), conds()]),
       y = one_of([forloops(), conds()]),
       c = one_of([floats(), integers()])) # complexes() doesn't work
@settings(max_examples=10)
def test_eval_fors_conds(x, y, c):
    def render(style):
        inner = saturates_poi1(ConstExpr(c), saturates_expr1(ConstExpr(c+c), y(style=style)))
        outer = saturates_expr1(ConstExpr(c+c+c), saturates_poi1(inner, x(style=style)))
        return POI.fE(outer)

    r1 = evalPOI_(render(CFS.Python), use_qjit=False)
    r2 = evalPOI_(render(CFS.Catalyst), use_qjit=True)
    assert_allclose(r1, r2)


@mark.parametrize('st', [CFS.Python, CFS.Catalyst])
@given(d=data())
@settings(verbosity=Verbosity.debug)
def test_pprint_fdef_cflow(d, st):
    x = d.draw(one_of([whileloops(style=st), forloops(style=st), conds(style=st)]))
    f = FDefStmt(FName("main"), [], POI())
    s = pstr(saturates_poi1(saturates_expr1(ConstExpr(42), saturates_poi1(ConstExpr(33), x())), f))
    note(s)


# @mark.parametrize('st', [CFS.Python, CFS.Catalyst])
# @given(d=data())
# @settings(verbosity=Verbosity.debug)
# def test_pprint_ret_ctflow(d, st):
#     x = d.draw(one_of([whileloops(style=st), forloops(style=st), conds(style=st)]))
#     s = pstr(RetStmt( saturates_expr1(ConstExpr(42), saturates_poi1(ConstExpr(33), x))))
#     note(s)


@given(x=one_of([conds(),whileloops(),forloops()]))
def test_eq_expr(x):
    xa = saturates_poi1(POI(), x())
    xb = saturates_poi1(POI(), x())
    assert xa is not xb
    assert xa == xb
    xc = saturates_poi1(saturates_poi1(POI(), x()), x())
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
    x2 = saturates_expr1(ConstExpr(jx), saturates_poi1(ConstExpr(jx), c()))
    assert jx == evalPOI_(POI.fE(x2), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=forloops(lvars=just(VName('i')),svars=just(VName('s'))))
@settings(max_examples=10)
def test_eval_for(l, x, use_qjit):
    jx = jnp.array([x])
    r = saturates_expr1(ConstExpr(jx), saturates_poi1(VRefExpr(VName('s')), l()))
    assert jx == evalPOI_(POI.fE(r), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=whileloops(lvars=just(VName('i')),
                    lexpr=lambda _: just(falseExpr)))
@settings(max_examples=10)
def test_eval_while(l, x, use_qjit):
    jx = jnp.array([x])
    r = saturates_poi1(addExpr(VRefExpr(VName('i')), ConstExpr(1)),
                       saturates_expr1(ConstExpr(jx), l()))
    assert jx == evalPOI_(POI.fE(r), use_qjit=use_qjit)


@given(g=qgates, m=qmeasurements)
@settings(max_examples=100, verbosity=Verbosity.debug)
def test_eval_qops(g, m):
    evalPOI_(POI([AssignStmt.fE(g)],m),
             use_qjit=True,
             qnode_device="lightning.qubit",
             qnode_wires=1)


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
    b.update(0, POI.fE(saturate_expr1(l, ConstExpr(0))))
    b.update(1, POI.fE(saturate_expr1(c, ConstExpr(1))))
    assert len(b.pois)==4
    s1 = pstr_builder(b)
    b.update(0, b.pois[0].poi)
    assert len(b.pois)==4
    s2 = pstr_builder(b)
    assert s1 == s2


def test_build_assign_layout():
    va = AssignStmt(VName('a'),ConstExpr(33))
    vb = AssignStmt(VName('b'),ConstExpr(42))
    l = WhileLoopExpr(VName("i"), trueExpr, POI([vb],VRefExpr(VName('b'))), ControlFlowStyle.Catalyst)
    b = build(POI([va],saturate_expr1(l, ConstExpr(0))))
    s = pstr_builder(b)
    print(b.pois[0].ctx)


@mark.parametrize('qnode_device', [None, "lightning.qubit"])
@mark.parametrize('use_qjit', [True, False])
@mark.parametrize('scalar', [0, -2.32323e10])
def test_run(use_qjit, qnode_device, scalar):
    val = jnp.array(scalar)
    source_file = mktemp("source.py")
    code, res = runPOI(POI.fE(ConstExpr(val)), use_qjit=use_qjit, qnode_device=qnode_device,
                       source_file=source_file)
    os.remove(source_file)
    assert res is not None
    assert_allclose(val, res)



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

# sample_spec:Dict[Expr,int] = {
#     WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst) : 1,
#     ForLoopExpr(VName("i"), ConstExpr(0), ConstExpr(10), POI(), ControlFlowStyle.Catalyst) : 2,
#     # CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst) : 1,
# }

# def run_greedy():
#     for b in control_flows(sample_spec):
#         pprint(b)


sample_spec:List[Expr] = [
    # WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    WhileLoopExpr(VName("j1"), lessExpr(VRefExpr(VName("j1")),ConstExpr(2)), POI(), CFS.Python),
    ForLoopExpr(VName("k1"), ConstExpr(0), ConstExpr(2), POI(), CFS.Python, VName("k2")),
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
]

gate_lib = [
    (FName("qml.Hadamard"), Signature(['*'],'*')),
    (FName("qml.X"), Signature(['*'],'*')),
]

def bindAssign(poi1:POI, fpoi2:Callable[[Expr],POI]):
    poi2 = fpoi2(poi1.expr)
    return bind(poi1, poi2, poi2.expr)


def run():
    arg = VName('arg')
    for b in control_flows(sample_spec, gate_lib, [arg]):
        print("1. Builder:")
        pprint(b)
        print("1. Press Enter to compile")
        input()
        o,code = compilePOI(
            bindAssign(b.pois[0].poi,
                       lambda e: POI([AssignStmt(None,e)],FCallExpr(VRefExpr(FName("qml.state")),[]))),
            use_qjit=True, name="main", qnode_wires=3, args=[arg])
        print("2. Compiled code:")
        print(code)
        print("2. Press Enter to eval")
        input()
        # r = evalPOI(o, name="main", args=[(arg,0)])
        # print("3. Evaluation result:")
        # print(r)



