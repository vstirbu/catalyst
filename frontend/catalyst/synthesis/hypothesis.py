from typing import Dict, List, Any, Optional
from functools import partial
from copy import deepcopy

from hypothesis.strategies import (text, decimals, integers, characters, from_regex, dictionaries,
                                   one_of, lists, recursive, none, booleans, floats, composite,
                                   binary, sets, permutations, sampled_from, data, from_regex,
                                   uuids, just)

from .grammar import (VName, FName, FDefStmt, CondExpr, ForLoopExpr, POI, WhileLoopExpr, trueExpr,
                      falseExpr, ControlFlowStyle, ConstExpr, Expr, VRefExpr, bindUnary, signature,
                      NoneExpr)
from .builder import build
from .pprint import pprint

@composite
def complexes(draw, re=None, im=None, **kwargs):
    re=re if re else floats(**kwargs)
    im=im if im else floats(**kwargs)
    return complex(draw(re),draw(im))

@composite
def symbols(draw, prefixes, suffixes=integers(0,10))->str:
    return f"{draw(prefixes)}{draw(suffixes)}"

@composite
def vnames(draw, prefixes=just("var"))->VName:
    return VName(draw(symbols(prefixes)))

@composite
def fnames(draw, prefixes=just("fun"))->FName:
    return FName(draw(symbols(prefixes)))

@composite
def fdefs(draw,
          names=fnames(just("fun")),
          args=vnames(just("arg"))):
    return partial(FDefStmt,
                   fname=draw(name),
                   ags=draw(lists(args,max_size=4,unique=True)))

@composite
def conds(draw,
          cond=sampled_from([trueExpr, falseExpr]),
          style=ControlFlowStyle.Catalyst):
    return partial(CondExpr, cond=draw(cond), style=style)


@composite
def forloops(draw,
             lvars=vnames(sampled_from(['i','j','k'])),
             lbounds=integers(0,10),
             ubounds=integers(0,10),
             style=ControlFlowStyle.Catalyst):
    return partial(ForLoopExpr,
                   loopvar=draw(lvars),
                   lbound=ConstExpr(draw(lbounds)),
                   ubound=ConstExpr(draw(ubounds)),
                   style=style)

@composite
def whileloops(draw,
               lname=vnames(sampled_from(['i','j','k'])),
               lexpr=sampled_from([trueExpr, falseExpr]),
               style=ControlFlowStyle.Catalyst):
    return partial(WhileLoopExpr,
                   loopvar=draw(lname),
                   cond=draw(lexpr),
                   style=style)


@composite
def programs(draw, spec:Dict[Expr,int], vscope:List[VName]):
    spec = deepcopy(spec)
    b = build(POI.fromExpr(VRefExpr(draw(sampled_from(vscope)))), vscope)
    print(b)
    while sum(spec.values())>0:
        options = [k for k,v in spec.items() if v>0]
        print('O', options)
        e = draw(sampled_from(options))
        p = draw(sampled_from(range(len(b.pois))))
        combined = bindUnary(b.at(p).poi, POI.fromExpr(deepcopy(e)))
        print(combined)
        if combined:
            pwcs = b.update(p, combined)
            for pwc in [pwc for pwc in pwcs if pwc.poi.expr == NoneExpr()]:
                vname = draw(sampled_from(pwc.ctx.get_vscope()))
                b.update(pwc, POI.fromExpr(VRefExpr(vname)), assert_no_delete=True)
            spec[e]-=1
        else:
            s = signature(e)
            print(s)

    # pprint(b)
    return b

