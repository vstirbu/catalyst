from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt, CondExpr,
                      WhileLoopExpr, FDefStmt, Program, RetStmt, ConstExpr, POI, ForLoopExpr,
                      WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle as CFS, assert_never,
                      NoneExpr, saturate_expr1, addExpr, lessExpr, Signature,
                      AssignStmt, signature, get_vars)

from .builder import Builder, contextualize_expr, build
from .pprint import pprint

def npois(e:Expr) -> int:
    return len(contextualize_expr(e))


def expanded_to(ls:List[Any], l:int)->List[Optional[Any]]:
    return [(ls[i] if i<len(ls) else None) for i in range(max(len(ls),l))]


def control_flows(expr_lib:List[Expr],
                  gate_lib:List[Tuple[FName,Signature]],
                  free_vars:List[VName]=[]) -> Iterable[Builder]:
    gs = gate_lib
    es = expr_lib
    ps = sum([npois(e) for e in expr_lib], 1)
    vs = sum([get_vars(e) for e in expr_lib], free_vars)
    nargs = max(len(s.args) for _,s in gate_lib) + max(len(signature(e).args) for e in expr_lib)
    for e_sample in permutations(es):
        for p_sample in permutations(range(ps)):
            for g_sample in product(*[gs]*len(p_sample)):
                args = list(product(*([vs]*nargs)))
                for v_sample in product(*([args]*len(p_sample))):
                    b = build(POI(), free_vars)
                    try:
                        for p,g,e,v in zip(p_sample,
                                           g_sample,
                                           expanded_to(e_sample,len(v_sample)),
                                           v_sample):
                            ctx = b.at(p).ctx
                            assert len(v) == 2
                            assert all(vi in ctx.get_vscope() for vi in v), \
                                f"{v} not in scope: {ctx.get_vscope()}"
                            n = saturate_expr1(deepcopy(e) if e else VRefExpr(v[1]), VRefExpr(v[1]))
                            r = addExpr(VRefExpr(ctx.statevar), n) if ctx.statevar else n
                            b.update(p,
                                     POI([AssignStmt(None, FCallExpr(VRefExpr(g[0]),
                                                                     [VRefExpr(v[0])]))], r))
                        yield b
                    except AssertionError as e:
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

