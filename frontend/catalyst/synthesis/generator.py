from typing import Union, List, Optional, NoReturn, Callable, Tuple, Any, Set
from dataclasses import dataclass
from copy import deepcopy

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondStmt, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POIStmt, ForLoopStmt,
                      ControlFlowStyle as CFS, assert_never)

import numpy as np
from numpy import floor
from numpy.random import seed as np_seed, rand, uniform, set_state, get_state

class RNGBase:
    """ Base interface class for Random Number Generators """
    def sample(self) -> float:
        raise NotImplementedError
    def sample_uniform(self, upper:int, lower:int=0) -> int:
        raise NotImplementedError

class RNG(RNGBase):
    """ Sample numpy RNG implementation, not for parallel execution. """
    def __init__(self, seed:int):
        np_seed(seed)
        self.state = get_state()
    def sample(self) -> float:
        set_state(self.state)
        r = rand()
        self.state = get_state()
        return r
    def sample_uniform(self, upper:int, lower:int=0) -> int:
        set_state(self.state)
        r = int(floor(uniform(lower, upper)))
        self.state = get_state()
        return r


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
class Context:
    """ Context of POI contains some information required to insert new
    statemtents at POI. """
    vscope: Set[VName]
    nwires: Optional[int]
    parent: Optional["Context"]

    def __init__(self,
                 vscope:Optional[Set[VName]]=None, nwires=None, parent=None):
        self.vscope = vscope if vscope is not None else set()
        self.nwires = nwires
        self.parent = parent

    def get_vscope(self) -> Set[VName]:
        return self.vscope | (self.parent.get_vscope() if self.parent else set())

@dataclass
class POIWithContext:
    """ Point Of Insertion with the context tracks the information which is
    required for making POI insertions. """
    poi: POIStmt
    ctx: Context

PWC = POIWithContext
""" A shorter alias for POIWithContext """

@dataclass
class POITracker:
    """ POITracker maintains the program being built and keeps all its
    points of insertion acessible and their contexts are known. """
    stmt: Stmt
    pois: List[POIWithContext]

    def _resolve(self, poic:Union[int,PWC]) -> POIWithContext:
        if isinstance(poic, int):
            return self.pois[poic]
        elif isinstance(poic, POIWithContext):
            return poic
        else:
            assert_never(poic)

    def insert_statement(self, poic:Union[int,PWC], s:Stmt) -> "POITracker":
        """ Add a new statement at the point of insertion """
        poicI:PWC = self._resolve(poic)
        for poicS in contextualize(s):
            if poicS.ctx.parent is None:
                poicS.ctx.parent = poicI.ctx
            self.pois.append(poicS)
        poicI.poi.stmts.append(s)
        if isinstance(s, AssignStmt) and s.vname is not None:
            poicI.ctx.vscope.add(s.vname)
        return self

    def insert_tracker(self, poic:Union[int,PWC], sc2:"POITracker") -> "POITracker":
        return self.insert_statement(poic, sc2.stmt)

def sample_fdef(rng:RNGBase, fname:FName, args:List[VName],
                qwires=None) -> FDefStmt:
    return FDefStmt(fname, args, POIStmt(), qwires=qwires)

def sample_while(rng:RNGBase, cond:Expr) -> WhileLoopStmt:
    return WhileLoopStmt(cond, POIStmt())

def sample_cond(rng:RNGBase, cond:Expr, style=CFS.Python) -> CondStmt:
    return CondStmt(cond, POIStmt(),
                    sample_oneof(rng, [None, POIStmt()]), style)

def pois_scan_inplace(ss:List[Stmt], ctx:Context, acc:List[PWC]) -> None:
    for s in ss:
        if isinstance(s, AssignStmt) and s.vname is not None:
            ctx.vscope.add(s.vname)
        acc.extend(contextualize(s, ctx))

def contextualize(s:Stmt, ctx:Optional[Context]=None) -> List[PWC]:
    """ Recursively collect insertion contexts across the statement. """
    acc:List[PWC] = list()
    if isinstance(s, AssignStmt):
        return []
    elif isinstance(s, WhileLoopStmt):
        ctx1 = Context(parent=ctx)
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(PWC(s.body, ctx1))
        return acc
    elif isinstance(s, ForLoopStmt):
        ctx1 = Context(parent=ctx, vscope=set([s.loopvar]))
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(PWC(s.body, ctx1))
        return acc
    elif isinstance(s, CondStmt):
        ctx1 = Context(parent=ctx)
        pois_scan_inplace(s.trueBranch.stmts, ctx1, acc)
        acc.append(POIWithContext(s.trueBranch, ctx1))
        if s.falseBranch is not None:
            ctx2 = Context(parent=ctx)
            pois_scan_inplace(s.falseBranch.stmts, ctx2, acc)
            acc.append(POIWithContext(s.falseBranch, ctx2))
        return acc
    elif isinstance(s, FDefStmt):
        ctx1 = Context(set(s.args),parent=ctx)
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(POIWithContext(s.body, ctx1))
        return acc
    else:
        assert_never(s)

def track(s:Stmt) -> POITracker:
    """ Construct the statement insertion tracker """
    return POITracker(s, contextualize(s))



