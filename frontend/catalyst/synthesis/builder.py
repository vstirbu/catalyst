from typing import Union, List, Optional, NoReturn, Callable, Tuple, Any, Set
from dataclasses import dataclass
from copy import deepcopy

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopExpr, FDefStmt, Program, RetStmt,
                      ConstExpr, POI, ForLoopExpr,
                      ControlFlowStyle as CFS, assert_never)

import numpy as np
from numpy import floor
from numpy.random import seed as np_seed, rand, uniform, set_state, get_state


@dataclass
class Context:
    """ Context of POI contains some information required to insert new
    statemtents at POI. """
    vscope: List[VName]
    nwires: Optional[int]
    parent: Optional["Context"]

    def __init__(self,
                 vscope:Optional[List[VName]]=None, nwires=None, parent=None):
        self.vscope = vscope if vscope is not None else []
        self.nwires = nwires
        self.parent = parent

    def get_vscope(self) -> List[VName]:
        return self.vscope + (self.parent.get_vscope() if self.parent else [])

    def parents(self) -> list:
        return [] if self.parent is None else [self.parent] + self.parent.parents()

def is_parent_of(parent, child) -> bool:
    return id(parent) in map(id,child.parents())


@dataclass(frozen=True)
class POIWithContext:
    """ Point Of Insertion with the context tracks the information which is
    required for making POI insertions. """
    poi: POI
    ctx: Context

PWC = POIWithContext
""" A shorter alias for POIWithContext """

PRef = Union[int, PWC]
""" Index of PWC: either an integer or the referece """

@dataclass
class Builder:
    """ Builder maintains the program being built and keeps all its
    points of insertion acessible and their contexts are known. """
    # root: POI
    pois: List[POIWithContext]

    def at(self, n:PRef) -> PWC:
        if isinstance(n, int):
            return self.pois[n]
        elif isinstance(n, PWC):
            assert n in self.pois
            return n
        else:
            raise ValueError("Invalid value passed to Builder.at() as a key")

    def vscope_at(self, n:PRef) -> List[VName]:
        return self.at(n).ctx.get_vscope()

    def update(self, n:PRef, poi:POI, assert_no_delete=False) -> List[PWC]:
        """ Add a new statement at the point of insertion, return the list of new PWCs """
        poic:PWC = self.at(n)
        for i in reversed(range(len(self.pois))):
            if is_parent_of(poic.ctx, self.pois[i].ctx):
                # assert i != n, f"We surely don't delete the requested POI #{n}"
                print(f"Removing {i}")
                assert not assert_no_delete, f"But we do delete {i} when updating {n}!"
                del self.pois[i]
        pwcs,_ = contextualize_poi(poi, poic.ctx)
        # print(f"Added {len(pwcs)} more")
        self.pois.extend(pwcs)
        poic.poi.stmts = poi.stmts
        poic.poi.expr = poi.expr
        for pwc in pwcs:
            assert is_parent_of(poic.ctx, pwc.ctx)
            assert pwc in self.pois
        return pwcs


def pois_scan_inplace(ss:List[Stmt], ctx:Context, acc:List[PWC]) -> Context:
    for s in ss:
        if isinstance(s, AssignStmt) and s.vname is not None:
            ctx = Context([s.vname], parent=ctx)
        acc.extend(contextualize_stmt(s, ctx))
    return ctx


def contextualize_expr(e:Expr, ctx:Optional[Context]=None) -> List[PWC]:
    acc:List[PWC] = list()
    if isinstance(e, CondExpr):
        ctx1 = Context(parent=ctx)
        ctx1 = pois_scan_inplace(e.trueBranch.stmts, ctx1, acc)
        acc.append(POIWithContext(e.trueBranch, ctx1))
        acc.extend(contextualize_expr(e.trueBranch.expr, ctx1))
        if e.falseBranch is not None:
            ctx2 = Context(parent=ctx)
            ctx2 = pois_scan_inplace(e.falseBranch.stmts, ctx2, acc)
            acc.append(POIWithContext(e.falseBranch, ctx2))
            acc.extend(contextualize_expr(e.falseBranch.expr, ctx2))
    elif isinstance(e, ForLoopExpr):
        acc.extend(contextualize_expr(e.lbound, ctx))
        acc.extend(contextualize_expr(e.ubound, ctx))
        ctx1 = Context(parent=ctx, vscope=[e.loopvar])
        ctx1 = pois_scan_inplace(e.body.stmts, ctx1, acc)
        acc.append(PWC(e.body, ctx1))
        acc.extend(contextualize_expr(e.body.expr, ctx1))
    elif isinstance(e, WhileLoopExpr):
        acc.extend(contextualize_expr(e.cond, ctx))
        ctx1 = Context(parent=ctx, vscope=[e.loopvar])
        ctx1 = pois_scan_inplace(e.body.stmts, ctx1, acc)
        acc.append(PWC(e.body, ctx1))
        acc.extend(contextualize_expr(e.body.expr, ctx1))
    elif isinstance(e, FCallExpr):
        acc.extend(contextualize_expr(e.expr, ctx))
        for a in e.args:
            acc.extend(contextualize_expr(a, ctx))
    else:
        pass
    return acc

def contextualize_stmt(s:Stmt, ctx:Optional[Context]=None) -> List[PWC]:
    """ Recursively collect insertion contexts across the statement. """
    acc:List[PWC] = list()
    if isinstance(s, AssignStmt):
        acc.extend(contextualize_expr(s.expr, ctx))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            acc.extend(contextualize_expr(s.expr, ctx))
    elif isinstance(s, FDefStmt):
        ctx1 = Context(s.args,parent=ctx)
        ctx1 = pois_scan_inplace(s.body.stmts, ctx1, acc)
        acc.append(POIWithContext(s.body, ctx1))
    else:
        assert_never(s)

    return acc

def contextualize_poi(poi:POI, ctx:Context) -> Tuple[List[PWC],Context]:
    pwc1 = list()
    ctx = pois_scan_inplace(poi.stmts, ctx, pwc1)
    pwc2 = contextualize_expr(poi.expr, ctx)
    return (pwc1 + pwc2, ctx)


def build(poi:POI, vscope:Optional[List[VName]]=None) -> Builder:
    ctx = Context(vscope)
    pwcs,ctx = contextualize_poi(poi, ctx)
    return Builder([POIWithContext(poi,ctx)] + pwcs)



