from typing import Dict, List, Any, Optional
from functools import partial
from copy import deepcopy

from hypothesis.strategies import (text, decimals, integers, characters, from_regex, dictionaries,
                                   one_of, lists, recursive, none, booleans, floats, composite,
                                   binary, sets, permutations, sampled_from, data, from_regex,
                                   uuids, just, data)

from .grammar import (VName, FName, FDefStmt, CondExpr, ForLoopExpr, POI, WhileLoopExpr, trueExpr,
                      falseExpr, ControlFlowStyle, ConstExpr, Expr, VRefExpr, bindUnary, signature,
                      NoneExpr, FName, FDefStmt, FCallExpr, lessExpr, eqExpr)

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
    fname=draw(name)
    ags=draw(lists(args,max_size=4,unique=True))
    return (lambda a:
            FDefStmt(
                fname=fname,
                args=args,
                body=a))

@composite
def conds(draw,
          cond=sampled_from([trueExpr, falseExpr]),
          style=ControlFlowStyle.Catalyst):
    cond=draw(cond)
    return partial(CondExpr,
                cond=cond,
                trueBranch=POI(),
                falseBranch=POI(),
                style=style)


@composite
def forloops(draw,
             lvars=vnames(sampled_from('ijk')),
             svars=vnames(sampled_from('lmn')),
             lbounds=integers(0,10),
             ubounds=integers(0,10),
             style=ControlFlowStyle.Catalyst):
    loopvar=draw(lvars)
    statevar=draw(svars)
    lbound=ConstExpr(draw(lbounds))
    ubound=ConstExpr(draw(ubounds))
    return partial(ForLoopExpr,
                loopvar=loopvar,
                statevar=statevar,
                lbound=lbound,
                ubound=ubound,
                body=POI(),
                style=style)

@composite
def whileloops(draw,
               lvars=vnames(sampled_from('ijk')),
               lexpr=lambda x: just(eqExpr(x,ConstExpr(0))),
               style=ControlFlowStyle.Catalyst):
    loopvar=draw(lvars)
    cond=draw(lexpr(VRefExpr(loopvar)))
    return partial(WhileLoopExpr,
                loopvar=loopvar,
                cond=cond,
                body=POI(),
                style=style)

qml_X = FCallExpr(VRefExpr(FName("qml.X")),[ConstExpr(0)])
qml_H = FCallExpr(VRefExpr(FName("qml.Hadamard")),[ConstExpr(0)])
qml_state = FCallExpr(VRefExpr(FName("qml.state")),[])
qgates = sampled_from([qml_X, qml_H])
qmeasurements = sampled_from([qml_state])


@composite
def programs(draw, spec:Dict[Expr,int], vscope:List[VName]):
    spec = deepcopy(spec)
    b = build(POI([FDefStmt(FName("main"), vscope, POI.fromExpr(VRefExpr(VName('x'))))]))
    print(b)
    while sum(spec.values())>0:
        options = [k for k,v in spec.items() if v>0]
        print('O', options)
        e = draw(sampled_from(options))
        p = draw(sampled_from(range(1,len(b.pois))))
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

