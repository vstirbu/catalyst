import sys
import math
import cmath
from typing import Any, Dict, Tuple, Union
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


def compileExpr(e :Expr, use_qjit:bool=True) -> Tuple[PythonObj, PythonCode]:
    main = FDefStmt(FName("main"), [], POI.fromExpr(
        e if expr_signature(e) is None else mkCallExpr1(e, CondExpr(0))), qjit=use_qjit)
    code = '\n'.join(pstr_stmt(main))
    o = compile(code, "<compileExpr>", "single")
    return (o, code)


def evalExpr(e: Union[Expr,PythonObj], **kwargs) -> Any:
    o = compileExpr(e, **kwargs)[0] if isinstance_expr(e) else e
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

