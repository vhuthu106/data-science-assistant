"""Safety checks for generated code.

This is a GUARD RAIL, not a sandbox. It blocks everything a data analysis on an
already-loaded DataFrame should never need (files, network, the operating system,
Python internals). Real isolation would run the code in a separate process or
container with no file or network access.

Two layers:
  1. check_code():  read the code BEFORE running it (Python's own parser, `ast`)
  2. SAFE_BUILTINS: while it runs, dangerous built-in functions simply don't exist
"""
import ast
import builtins

# Libraries generated code may import. Everything else is refused.
ALLOWED_MODULES = {"pandas", "numpy", "matplotlib", "seaborn", "scipy",
                   "math", "statistics", "collections", "itertools", "datetime", "re"}

# Built-in functions that give access to files, the interpreter, or arbitrary attributes.
BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit",
                    "help", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                    "memoryview", "__import__"}

# pandas/numpy/matplotlib methods that read or write files or URLs.
BLOCKED_METHODS = {"savefig", "imread", "imsave", "load", "save", "savez", "savez_compressed",
                   "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "memmap"}
# Libraries re-export system modules as attributes (e.g. pandas exposes `pd.io.common.os`),
# so dangerous modules must be blocked by NAME wherever they appear, not only in imports.
BLOCKED_ATTRIBUTES = {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "glob", "tempfile",
                      "builtins", "importlib", "pickle", "io", "urllib", "http", "requests", "ctypes",
                      "system", "popen", "environ"}
SAFE_TO_METHODS = {"to_numeric", "to_datetime", "to_timedelta", "to_frame", "to_list", "to_dict",
                   "to_numpy", "to_string", "to_period", "to_timestamp", "to_records", "to_markdown"}


def check_code(code: str) -> list[str]:
    """Return a list of problems (with line numbers). An empty list means the code may run."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # not dangerous; the executor reports the syntax error with its line number

    problems = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append(f"line {line}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.split(".")[0] not in ALLOWED_MODULES:
                problems.append(f"line {line}: import from '{module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_BUILTINS:
            problems.append(f"line {line}: '{node.id}' is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            problems.append(f"line {line}: '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if name.startswith("__") and name.endswith("__"):
                problems.append(f"line {line}: access to '{name}' is not allowed")
            elif name in BLOCKED_ATTRIBUTES:
                problems.append(f"line {line}: '.{name}' reaches the operating system and is not allowed")
            elif name.startswith("read_"):
                problems.append(f"line {line}: '{name}' reads files or URLs; use the loaded df")
            elif name.startswith("to_") and name not in SAFE_TO_METHODS:
                problems.append(f"line {line}: '{name}' writes files; put the answer in `result`")
            elif name in BLOCKED_METHODS:
                problems.append(f"line {line}: '{name}' reads or writes files")
    return problems


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Runtime twin of the import check, in case something slips past the static check."""
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {name: value for name, value in vars(builtins).items() if name not in BLOCKED_BUILTINS}
SAFE_BUILTINS["__import__"] = _guarded_import
