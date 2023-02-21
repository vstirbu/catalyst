from typing import Union, List, Optional, NoReturn, Callable, Tuple, Any, Set
from dataclasses import dataclass
from copy import deepcopy

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondStmt, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POI, ForLoopExpr,
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



def sample_fdef(rng:RNGBase, fname:FName, args:List[VName],
                qwires=None) -> FDefStmt:
    return FDefStmt(fname, args, POI(), qwires=qwires)

def sample_while(rng:RNGBase, cond:Expr) -> WhileLoopStmt:
    return WhileLoopStmt(cond, POI())

def sample_cond(rng:RNGBase, cond:Expr, style=CFS.Python) -> CondStmt:
    return CondStmt(cond, POI(),
                    sample_oneof(rng, [None, POI()]), style)

