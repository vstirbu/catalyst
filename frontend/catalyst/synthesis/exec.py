import sys
import math
import cmath
from typing import Any, Dict, Tuple, Union, Optional, List
from inspect import signature as python_signature, _empty as inspect_empty

import jax.numpy as jnp
import pennylane as qml
from catalyst import qjit, for_loop, while_loop, cond

from .grammar import (Expr, RetStmt, FDefStmt, FCallExpr, VName, FName, VRefExpr, signature as
                      expr_signature, isinstance_expr, POI)
from .pprint import pstr_stmt, pstr_expr, pprint
from .builder import build


PythonCode = str
PythonObj = Any


def compilePOI(p:POI,
               use_qjit:bool=True,
               name:Optional[str]=None,
               args:Optional[List[Expr]]=None) -> Tuple[PythonObj, PythonCode]:
    name = name if name is not None else "main"
    args = args if args is not None else []
    main = FDefStmt(FName(name), args, p, qjit=use_qjit)
    code = '\n'.join(pstr_stmt(main))
    o = compile(code, "<compilePOI>", "single")
    return (o, code)


def evalPOI(p: Union[POI,PythonObj], **kwargs) -> Any:
    o = compileExpr(p, **kwargs)[0] if isinstance(p,POI) else p
    gctx, lctx = {}, {}
    gctx.update({'qml': qml,
                 'for_loop':for_loop,
                 'while_loop':while_loop,
                 'cond':cond,
                 'qjit':qjit,
                 'Array':jnp.array,
                 'int64':jnp.int64,
                 'complex128':jnp.complex128,
                 'inf':math.inf,
                 'nan':math.nan,
                 'infj':cmath.infj,
                 'nanj':cmath.nanj})
    s = exec(o, gctx, lctx)
    r = eval("main()", gctx, lctx)
    return r

