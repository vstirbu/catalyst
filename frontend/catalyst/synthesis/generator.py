from typing import List, Optional, NoReturn, Callable, Tuple, Any, Set
from dataclasses import dataclass

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondStmt, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POIStmt, assert_never)

import numpy as np
from numpy.random import rand, uniform

class RNGBase:
    """ Base interface class for Random Number Generators """
    def sample(self) -> float:
        raise NotImplementedError
    def sample_uniform(self, upper:int, lower:int=0) -> int:
        raise NotImplementedError

class RNG(RNGBase):
    """ Sample numpy RNG implementation """
    def sample(self) -> float:
        return rand()
    def sample_uniform(self, upper:int, lower:int=0) -> int:
        return int(np.floor(uniform(lower, upper)))


def sample_oneof(rng:RNGBase, candidates:list) -> Any:
    return candidates[rng.sample_uniform(len(candidates))]

def sample_bools(rng:RNGBase) -> ConstExpr:
    return sample_oneof(rng, [ConstExpr(True), ConstExpr(False)])

def sample_ints(rng:RNGBase) -> ConstExpr:
    return sample_oneof(rng, [ConstExpr(0), ConstExpr(1), ConstExpr(42)])

def sample_vname(rng:RNGBase) -> VName:
    return VName(f"var{rng.sample_uniform(10)}")

def sample_fname(rng:RNGBase) -> FName:
    return FName(f"fun{rng.sample_uniform(10)}")

def sample_assign(rng:RNGBase, exprs:List[Expr]) -> AssignStmt:
    return AssignStmt(sample_vname(rng),
                      sample_oneof(rng, exprs))

@dataclass
class POI:
    """ Point Of Insertion referes to the point where we could insert new
    statements. POI tracks variables in scope. """
    root: POIStmt
    vscope: Set[VName]

@dataclass
class StmtCandidate:
    """ StmtCandidate maintains the statement being built and keeps all its
    points of insertion acessible. """
    stmt: Stmt
    pois: List[POI]

FDEF_ARG_ID:int = 0

def sample_fdef(rng:RNGBase, nargs:int) -> FDefStmt:
    global FDEF_ARG_ID
    args = [VName(f"arg{FDEF_ARG_ID+i}") for i in range(nargs)]
    FDEF_ARG_ID += nargs
    return FDefStmt([FName('qjit')], sample_fname(rng), args, POIStmt())

def sample_while(rng:RNGBase, cond:Expr) -> WhileLoopStmt:
    return WhileLoopStmt(cond, POIStmt())

def sample_cond(rng:RNGBase, cond:Expr) -> CondStmt:
    return CondStmt(cond, POIStmt(), sample_oneof(rng, [None, POIStmt()]))

def pois_toplevel(s:Stmt) -> List[POI]:
    """ List the top-level POIs of a statement """
    def _c(ss, scope=None):
        return [POI(s, (set(scope) if scope else set())) for s in ss]
    if isinstance(s, AssignStmt):
        return _c([])
    elif isinstance(s, WhileLoopStmt):
        return _c([s.body])
    elif isinstance(s, CondStmt):
        return _c([s.trueBranch] + ([s.falseBranch] if s.falseBranch else []))
    elif isinstance(s, FDefStmt):
        return _c([s.body], s.args)
    else:
        assert_never(s)

def candidate(s:Stmt) -> StmtCandidate:
    return StmtCandidate(s, pois_toplevel(s))

def insert_inplace(sc1:StmtCandidate, poi1:POI, s:Stmt) -> StmtCandidate:
    """ """
    pois2 = [POI(poi.root, poi1.vscope|poi.vscope) for poi in pois_toplevel(s)]
    poi1.root.stmts.append(s)
    sc1.pois.extend(pois2)
    return sc1

def combine_inplace(sc1:StmtCandidate, poi1:POI, sc2:StmtCandidate) -> StmtCandidate:
    """ Combine two statement candidates into one by inserting the latter into
    the point of insertion `poi1` of the former."""
    poi1.root.stmts.append(sc2.stmt)
    scope1 = poi1.vscope
    pois2 = [POI(poi.root, scope1|poi.vscope) for poi in sc2.pois]
    sc1.pois.extend(pois2)
    return sc1


