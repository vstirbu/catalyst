from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopExpr, FDefStmt, Program, RetStmt,
                      ConstExpr, POI, ForLoopExpr, WhileLoopExpr, trueExpr, falseExpr,
                      ControlFlowStyle as CFS, assert_never, NoneExpr)
from .builder import Builder, contextualize_expr, build
from .pprint import pprint

sample_spec:Dict[Expr,int] = {
    WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
}

def npois(e:Expr) -> int:
    return len(contextualize_expr(e))

def greedy(spec:Dict[Expr,int]) -> Iterable[Builder]:
    ps = sum([npois(k)*v for k,v in spec.items()],0)
    es = sum([[k]*v for k,v in spec.items()], [])
    # print(ps)
    # print(es)
    for e_sample in permutations(es):
        # for p_sample in product(range(ps), repeat=len(e_sample)):
        for p_sample in permutations(range(ps)):
            # print(e_sample)
            # print(p_sample)
            b = build(POI())
            try:
                for p,e in zip(p_sample, e_sample):
                    poi,ctx = astuple(b.at(p))
                    b = b.update(p, POI.fromExpr(deepcopy(e)) )
                for n, pc in enumerate(b.pois):
                    if pc.poi.expr == NoneExpr():
                        vscope = pc.ctx.get_vscope()
                        if len(vscope)>0:
                            b.update(n, POI.fromExpr(VRefExpr(vscope[0])))
                yield b
            except Exception as e:
                print(f"Oops", e)
            input()



