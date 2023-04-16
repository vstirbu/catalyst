from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product, chain, cycle

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt, CondExpr,
                      WhileLoopExpr, FDefStmt, Program, RetStmt, ConstExpr, POI, ForLoopExpr,
                      WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle as CFS, assert_never,
                      NoneExpr, saturate_expr1, addExpr, lessExpr, Signature,
                      AssignStmt, signature, get_vars, assignStmt, assignStmt_, callExpr)

from .builder import Builder, contextualize_expr, build
from .pprint import pprint

def npois(e:Expr) -> int:
    return len([p for p in contextualize_expr(e) if p.poi.isempty()])


def expanded_to(ls:List[Any], l:int)->List[Optional[Any]]:
    return [(ls[i] if i<len(ls) else None) for i in range(max(len(ls),l))]


def control_flows(expr_lib:List[Expr],
                  gate_lib:List[Tuple[FName,Signature]],
                  free_vars:List[VName]=[]) -> Iterable[Builder]:
    gs = gate_lib if gate_lib else [None]
    es = expr_lib
    ps = sum([npois(e) for e in expr_lib], 1)
    vs = sum([get_vars(e) for e in expr_lib], free_vars)
    nargs = max(chain([0],(len(s.args) for _,s in gate_lib))) + \
            max(len(signature(e).args) for e in expr_lib)
    for e_sample in permutations(es):
        for p_sample in permutations(range(ps)):
            for g_sample in product(*[gs]*len(p_sample)):
                args = list(product(*([vs]*max(2,nargs))))
                for v_sample in product(*([args]*len(p_sample))):
                    b = build(POI(), free_vars)
                    try:
                        e_sample_ext = expanded_to(e_sample,len(v_sample))
                        for p,g,e,v in zip(p_sample,
                                           g_sample,
                                           e_sample_ext,
                                           v_sample):
                            ctx = b.at(p).ctx
                            assert len(v) >= 2, f"len({v}) < 2"
                            if not all(vi in ctx.get_vscope() for vi in v):
                                raise IndexError(f"{v} not in scope: {ctx.get_vscope()}")
                            stmts = [AssignStmt_(callExpr(g[0], [v[0]]))] if g else []
                            res = saturate_expr1(e if e else v[1], v[1])
                            expr = addExpr(VRefExpr(ctx.statevar), res) if ctx.statevar else res
                            b.update(p, POI(stmts, expr), ignore_nonempty=True)
                        yield b
                    except IndexError as err:
                        pass


sample_spec:List[Expr] = [
    WhileLoopExpr(VName("j"), lessExpr(VRefExpr(VName("j")),ConstExpr(1)), POI(), CFS.Default),
    ForLoopExpr(VName("k1"), ConstExpr(0), ConstExpr(1), POI(), CFS.Default, VName("k2")),
]

gate_lib = [
    (FName("qml.X"), Signature(['*'],'*')),
    (FName("qml.H"), Signature(['*'],'*'))
]


def run():
    for b in control_flows(sample_spec, gate_lib, [VName('X')]):
        pprint(b)
        input()

