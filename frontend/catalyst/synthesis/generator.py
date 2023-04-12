from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt, CondExpr,
                      WhileLoopExpr, FDefStmt, Program, RetStmt, ConstExpr, POI, ForLoopExpr,
                      WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle as CFS, assert_never,
                      NoneExpr, reduce_expr, saturate_expr, addExpr, lessExpr)
from .builder import Builder, contextualize_expr, build
from .pprint import pprint

def npois(e:Expr) -> int:
    return len(contextualize_expr(e))


def evars(e:Expr) -> List[VName]:
    def _vars(e):
        if isinstance(e, ForLoopExpr):
            return [e.loopvar]
        elif isinstance(e, WhileLoopExpr):
            return [e.loopvar]
        elif isinstance(e, VRefExpr):
            return [e.vname]
        else:
            return []
    return reduce_expr(e, lambda e,acc: acc+_vars(e), [])


def expanded_to(ls:List[Any], l:int)->List[Optional[Any]]:
    return [(ls[i] if i<len(ls) else None) for i in range(max(len(ls),l))]


def control_flows(spec:Dict[Expr,int], free_vars:List[VName]=[]) -> Iterable[Builder]:
    """
    TODO: Output programs are not unique (not sure if one needs to change it).
    """
    es = sum([[k]*v for k,v in spec.items()], [])
    ps = sum([npois(k)*v for k,v in spec.items()], 1)
    vs = sum([evars(k)*v for k,v in spec.items()], free_vars)
    print('------> es')
    print('\n'.join(map(str,es)))
    print('------> ps')
    print('\n'.join(map(str,range(ps))))
    print('------> vs')
    print('\n'.join(map(str,vs)))
    for e_sample in permutations(es):
        for p_sample in permutations(range(ps)):
            for v_sample in product(*([vs]*len(p_sample))):
                print('==========')
                print('\n'.join(map(str,expanded_to(e_sample,len(v_sample)))))
                print('\n'.join(map(str,p_sample)))
                print('\n'.join(map(str,v_sample)))
                b = build(POI(), free_vars)
                try:
                    for p,e,v in zip(p_sample,
                                     expanded_to(e_sample,len(v_sample)),
                                     v_sample):
                        assert v in b.vscope_at(p), f"{v} is not in scope: {b.vscope_at(p)}"
                        tail = addExpr(VRefExpr(v),ConstExpr(1))
                        b.update(p,
                                 POI.fromExpr(
                                     saturate_expr(deepcopy(e), tail) if e else tail))
                    yield b
                except Exception as e:
                    pass
                    # print(f"Oops", e)

sample_spec:Dict[Expr,int] = {
    # WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    WhileLoopExpr(VName("j"), lessExpr(VRefExpr(VName("j")),ConstExpr(1)), POI(), CFS.Catalyst) : 1,
    ForLoopExpr(VName("k1"), ConstExpr(0), ConstExpr(1), POI(), CFS.Catalyst, VName("k2")) : 1,
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
}


def run():
    for b in control_flows(sample_spec, [VName('X')]):
        pprint(b)
        input()

