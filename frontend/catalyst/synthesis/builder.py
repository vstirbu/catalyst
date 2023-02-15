from typing import Union, List, Optional, NoReturn, Callable, Tuple, Any, Set
from dataclasses import dataclass
from copy import deepcopy

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopStmt, FDefStmt, Program, RetStmt,
                      ConstExpr, POI, ForLoopStmt,
                      ControlFlowStyle as CFS, assert_never)

import numpy as np
from numpy import floor
from numpy.random import seed as np_seed, rand, uniform, set_state, get_state


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
    poi: POI
    ctx: Context

PWC = POIWithContext
""" A shorter alias for POIWithContext """

@dataclass
class Builder:
    """ Builder maintains the program being built and keeps all its
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

    def insert_statement(self, poic:Union[int,PWC], s:Stmt) -> "Builder":
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

    def insert_tracker(self, poic:Union[int,PWC], sc2:"Builder") -> "Builder":
        return self.insert_statement(poic, sc2.stmt)

def pois_scan_inplace(ss:List[Stmt], ctx:Context, acc:List[PWC]) -> None:
    for s in ss:
        if isinstance(s, AssignStmt) and s.vname is not None:
            ctx.vscope.add(s.vname)
        acc.extend(contextualize(s, ctx))

def contextualize_expr(e:Expr, ctx:Optional[Context]=None) -> List[PWC]:
    acc:List[PWC] = list()
    if isinstance(e, CondExpr):
        ctx1 = Context(parent=ctx)
        pois_scan_inplace(e.trueBranch.stmts, ctx1, acc)
        acc.append(POIWithContext(e.trueBranch, ctx1))
        if e.falseBranch is not None:
            ctx2 = Context(parent=ctx)
            pois_scan_inplace(e.falseBranch.stmts, ctx2, acc)
            acc.append(POIWithContext(e.falseBranch, ctx2))
    else:
        pass
    return acc

def contextualize(s:Stmt, ctx:Optional[Context]=None) -> List[PWC]:
    """ Recursively collect insertion contexts across the statement. """
    acc:List[PWC] = list()
    if isinstance(s, AssignStmt):
        acc.extend(contextualize_expr(s.expr, ctx))
    elif isinstance(s, WhileLoopStmt):
        acc.extend(contextualize_expr(s.cond, ctx))
        ctx1 = Context(parent=ctx)
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(PWC(s.body, ctx1))
    elif isinstance(s, ForLoopStmt):
        acc.extend(contextualize_expr(s.lbound, ctx))
        acc.extend(contextualize_expr(s.ubound, ctx))
        ctx1 = Context(parent=ctx, vscope=set([s.loopvar]))
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(PWC(s.body, ctx1))
    # elif isinstance(s, CondStmt):
    #     ctx1 = Context(parent=ctx)
    #     pois_scan_inplace(s.trueBranch.stmts, ctx1, acc)
    #     acc.append(POIWithContext(s.trueBranch, ctx1))
    #     if s.falseBranch is not None:
    #         ctx2 = Context(parent=ctx)
    #         pois_scan_inplace(s.falseBranch.stmts, ctx2, acc)
    #         acc.append(POIWithContext(s.falseBranch, ctx2))
    #     return acc
    elif isinstance(s, FDefStmt):
        ctx1 = Context(set(s.args),parent=ctx)
        pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(POIWithContext(s.body, ctx1))
    else:
        assert_never(s)

    return acc

def build(s:Stmt) -> Builder:
    """ Construct the statement insertion tracker """
    s1 = deepcopy(s)
    return Builder(s1, contextualize(s1))



