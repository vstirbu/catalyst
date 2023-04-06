from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopExpr, FDefStmt, Program, RetStmt,
                      ConstExpr, POI, ForLoopExpr, WhileLoopExpr, trueExpr, falseExpr,
                      ControlFlowStyle as CFS, assert_never, NoneExpr, reduce_expr, saturate_expr)
from .builder import Builder, contextualize_expr, build
from .pprint import pprint

sample_spec:Dict[Expr,int] = {
    WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    WhileLoopExpr(VName("j"), trueExpr, POI(), CFS.Catalyst) : 1,
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
}

def npois(e:Expr) -> int:
    return len(contextualize_expr(e))

def evars(e:Expr) -> List[VName]:
    def _vars(e):
        if isinstance(e, ForLoopExpr):
            return [e.loopvar]
        elif isinstance(e, WhileLoopExpr):
            return [e.loopvar]
        elif isinstance(e, VRefExpr):
            return [e.loopvar]
        else:
            return []
    return reduce_expr(e, lambda e,acc: acc+_vars(e), [])


def control_flows(spec:Dict[Expr,int], free_vars:List[VName]=[]) -> Iterable[Builder]:
    """
    TODO: Implement `saturate_poi()`
    TODO: Output programs are not unique (not sure if one needs to change it).
    """
    ps = sum([npois(k)*v for k,v in spec.items()], 0)
    es = sum([[k]*v for k,v in spec.items()], [])
    vs = sum([evars(k)*v for k,v in spec.items()], free_vars)
    # print('vs',vs)
    for e_sample in permutations(es):
        for p_sample in permutations(range(ps)):
            for v_sample in product(*([vs]*len(p_sample))):
                # print('p_sample',p_sample)
                # print('v_sample',v_sample)
                b = build(POI(), free_vars)
                try:
                    for p,e,v in zip(p_sample, e_sample, v_sample):
                        assert v in b.vscope_at(p), f"{v} is not in scope: {b.vscope_at(p)}"
                        b.update(p, POI.fromExpr(saturate_expr(deepcopy(e), VRefExpr(v))))
                    yield b
                except Exception as e:
                    pass
                    # print(f"Oops", e)


def run():
    for b in control_flows(sample_spec, [VName('X')]):
        pprint(b)
        input()

