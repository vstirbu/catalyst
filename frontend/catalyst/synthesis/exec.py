import os
import sys
import math
import cmath
import subprocess
from typing import Any, Dict, Tuple, Union, Optional, List
from inspect import signature as python_signature, _empty as inspect_empty
from tempfile import mktemp
from os.path import basename

import jax
import jax.numpy as jnp
import pennylane.numpy as pnp
from pennylane.numpy import tensor as PnpArray
import pennylane as qml
try:
    from catalyst import qjit, for_loop, while_loop, cond
    CATALYST_LOADED = True
except ImportError:
    CATALYST_LOADED = False

from .grammar import (Expr, Stmt, RetStmt, FDefStmt, FCallExpr, VName, FName, VRefExpr, signature as
                      expr_signature, isinstance_expr, POI, ControlFlowStyle)
from .pprint import pstr_stmt, pstr_expr, pprint, PStrOptions
from .builder import build


PythonCode = str
PythonObj = Any


def wrapInMain(p:POI, name:Optional[str]=None, args:Optional[List[Expr]]=None, use_qjit:bool=True,
               **kwargs) -> Stmt:
    name = name if name is not None else "main"
    args = args if args is not None else []
    return FDefStmt(FName(name), args, p, qjit=use_qjit, **kwargs)


def compilePOI(p:Union[Stmt,POI],
               name:Optional[str]=None,
               args:Optional[List[Expr]]=None,
               default_cfstyle=ControlFlowStyle.Catalyst,
               **kwargs) -> Tuple[PythonObj, PythonCode]:
    """ Insert the point of insertion into the top-level function and use the Python built-in
    `compile` function on it """
    p = wrapInMain(p, name, args, **kwargs) if isinstance(p, POI) else p
    opts = PStrOptions(default_cfstyle)
    code = '\n'.join(pstr_stmt(p, None, opts))
    o = compile(code, "<compilePOI>", "single")
    return (o, code)


def pyenv(with_catalyst:bool=True) -> dict:
    if with_catalyst and not CATALYST_LOADED:
        raise RuntimeError("`Catalyst` is required (try `pip install catalyst`)")
    acc = {}
    acc.update({
        'qml':qml,
        'inf':math.inf,
        'nan':math.nan,
        'infj':cmath.infj,
        'nanj':cmath.nanj})

    if with_catalyst:
        acc.update({
            'np':jnp,
            'jax':jax,
            'for_loop':for_loop,
            'while_loop':while_loop,
            'cond':cond,
            'qjit':qjit,
            'Array':jnp.array,
            'int64':jnp.int64,
            'float64':jnp.float64,
            'complex128':jnp.complex128})
    else:
        acc.update({
            'np':pnp,
            'Array':pnp.tensor,
            'int64':pnp.int64,
            'float64':pnp.float64,
            'complex128':pnp.complex128})

    return acc

def pprint_pyenv(env:Optional[dict]=None, with_catalyst:bool=True) -> List[str]:
    env = env if env is not None else pyenv(with_catalyst)
    def _guess_module(obj):
        if hasattr(obj, "__module__"):
            return obj.__module__
        elif str(obj) in math.__dir__():
            return "math"
        elif str(obj) in cmath.__dir__():
            return "cmath"
        else:
            raise ValueError(f"Couldn't guess the module of {obj}")

    def _guess_name(obj, hint):
        if hasattr(obj, "__name__"):
            if obj.__name__ == 'IntdtypeSubclass':
                return hint
            else:
                return obj.__name__
        else:
            return str(obj)

    return [(f"import {v.__name__} as {k}" if "module" in str(type(v)) else
             f"from {_guess_module(v)} import {_guess_name(v,hint=k)} as {k}") for k,v in env.items()]



def evalPOI(p:Union[POI,PythonObj],
            args:Optional[List[Tuple[Expr,Any]]],
            name:Optional[str]=None,
            use_qjit:bool=True,
            **kwargs) -> Any:
    """ Evaluate the POI with Python built-in `eval` function in an isolated environment."""
    arg_exprs = list(zip(*args))[0] if args is not None and len(args)>0 else []
    arg_vals = list(zip(*args))[1] if args is not None and len(args)>0 else []
    name = name if name is not None else "main"
    o = compileExpr(p, args=arg_exprs, name=name, use_qjit=use_qjit, **kwargs)[0] if isinstance(p,POI) else p
    gctx, lctx = {}, {}
    gctx.update(pyenv(use_qjit))
    s = exec(o, gctx, lctx)
    r = eval(f"{name}({','.join(map(str,arg_vals))})", gctx, lctx)
    return r

def remove_noexept(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def runPOI(p:POI,
           name:Optional[str]=None,
           args:Optional[List[Tuple[Expr,Any]]]=None,
           source_file:Optional[str]=None,
           out_file:Optional[str]=None,
           use_qjit:bool=True,
           keep_intermediate:bool=False,
           interpreter:Optional[str]=None,
           timeout_sec:Optional[float]=5.0,
           **kwargs) -> Tuple[PythonCode,Optional[PnpArray]]:
    """ Run POI `p`, wrapped with a Python function as a stand-alone Python script. The POI is
    expected to return a numpy-compatible value. """
    name = name if name is not None else "main"
    args = args if args is not None else []
    arg_vals = list(zip(*args))[1] if args is not None and len(args)>0 else []
    main = FDefStmt(FName(name), args, p, qjit=use_qjit, **kwargs)
    interpreter = interpreter if interpreter is not None else sys.executable
    arg_prints = [pstr_expr(ConstExpr(a))[1] for a in arg_vals]

    source_file_, out_file_ = None, None
    try:
        source_file_ = source_file if source_file is not None else mktemp("source.py")
        out_file_ = out_file if out_file is not None else mktemp("out.npy")

        header = (
            ["import sys"] +
            pprint_pyenv(with_catalyst=use_qjit) +
            ["from argparse import ArgumentParser"] +
            (["jax.config.update('jax_enable_x64', True)",
              "jax.config.update('jax_platform_name', 'cpu')",
              "jax.config.update('jax_array', True)"] if use_qjit else [])
        )
        code = pstr_stmt(main)
        footer = (
            [f"AP = ArgumentParser(prog='python3 {basename(source_file_)}')",
             f"AP.add_argument('-o', '--output', type=str, default='_out.npy', metavar='FILE.npy', "
             "help='Output *.npy file')",
             f"np.save(AP.parse_args(sys.argv[1:]).output, main({','.join(arg_prints)}))" ])

        with open(source_file_, "w") as s:
            s.write('\n'.join(header + code + footer))

        remove_noexept(out_file_)
        cmdline = [interpreter,source_file_,'--output',out_file_]
        try:
            subprocess.check_output(cmdline, stderr=subprocess.STDOUT, timeout=timeout_sec)
            result = jnp.load(out_file_)
        except subprocess.TimeoutExpired:
            result = None
        return (code, result)
    finally:
        if source_file_ and source_file is None and not keep_intermediate:
            remove_noexept(source_file_)
        if out_file_ and out_file is None and not keep_intermediate:
            remove_noexept(out_file_)

