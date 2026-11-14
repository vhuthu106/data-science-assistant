"""Runs generated analysis code and collects everything it produced.

Kept separate from the LLM code so it can be tested on its own (no API key needed).

Conventions the generated code follows (the Worker's prompt tells the model):
  * `df`, `pd`, `np`, `plt`, `sns` already exist in the namespace
  * the main answer goes in a variable called `result`
  * charts are just drawn; the executor collects every open figure as a PNG
"""
import contextlib
import io
import traceback
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # draw into memory: a web server has no screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.safety import SAFE_BUILTINS, check_code

BASE_NAMES = {"df", "pd", "np", "plt", "sns", "__builtins__"}


@dataclass
class StepResult:
    code: str
    success: bool
    error: str | None = None                      # e.g. "KeyError (line 3): 'Agee'"
    output: str = ""                              # everything the code printed
    warnings: list[str] = field(default_factory=list)
    figures: list[bytes] = field(default_factory=list)   # PNG images
    result: object = None                         # value of `result`, if the code set it


def new_namespace(df: pd.DataFrame) -> dict:
    """One namespace shared by all steps of a plan, so step 2 sees step 1's changes."""
    return {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns,
            "__builtins__": SAFE_BUILTINS}   # open(), exec(), unsafe imports... don't exist here


def user_variables(namespace: dict) -> dict:
    """Variables created by earlier steps (name -> type name), to tell the model about them."""
    return {name: type(value).__name__ for name, value in namespace.items()
            if name not in BASE_NAMES | {"result"} and not name.startswith("_")}


def snapshot(namespace: dict) -> dict:
    """Copy of the namespace, so a failed step can't leave half-finished changes behind."""
    return {k: v.copy() if isinstance(v, (pd.DataFrame, pd.Series)) else v
            for k, v in namespace.items()}


def _collect_figures() -> list[bytes]:
    images = []
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        images.append(buffer.getvalue())
    plt.close("all")
    return images


def _describe_error(e: BaseException) -> str:
    """Short error message that points at the line of GENERATED code that failed."""
    if isinstance(e, SyntaxError):
        return f"SyntaxError (line {e.lineno}): {e.msg}"
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == "<generated>"]
    where = f" (line {lines[-1]})" if lines else ""
    return f"{type(e).__name__}{where}: {e}"


def run_code(code: str, namespace: dict) -> StepResult:
    """Execute `code` in `namespace` (modified in place) and collect its outputs."""
    problems = check_code(code)
    if problems:   # refused before running: nothing executed, nothing changed
        return StepResult(code, False, error="Blocked for safety: " + "; ".join(problems))
    plt.close("all")
    namespace.pop("result", None)          # never show a previous step's result by mistake
    stdout = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")    # report every warning, not just the first time
        try:
            compiled = compile(code, "<generated>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, namespace)  # ONE dict: functions in the code can see df, pd, ...
        except Exception as e:
            return StepResult(code, False, error=_describe_error(e), output=stdout.getvalue(),
                              warnings=[f"{w.category.__name__}: {w.message}" for w in caught],
                              figures=_collect_figures())
    return StepResult(code, True, output=stdout.getvalue(),
                      warnings=[f"{w.category.__name__}: {w.message}" for w in caught],
                      figures=_collect_figures(), result=namespace.get("result"))
