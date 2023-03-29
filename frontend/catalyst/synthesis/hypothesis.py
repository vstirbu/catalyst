from functools import partial

from hypothesis.strategies import (text, decimals, integers, characters, from_regex, dictionaries,
                                   one_of, lists, recursive, none, booleans, floats, composite,
                                   binary, sets, permutations, sampled_from, data, from_regex,
                                   uuids, just)

from catalyst.synthesis.grammar import (VName, FName, FDefStmt, CondExpr, ForLoopExpr, POI,
                                        WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle,
                                        ConstExpr)

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

