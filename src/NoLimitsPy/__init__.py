"""Python interface to NoLimits.jl.

Every exported NoLimits.jl name is available as ``NoLimitsPy.<name>`` via dynamic
forwarding; only the helpers defined here are hand-written. New NoLimits features
appear automatically with package updates - by design, do not add named wrappers.
"""

import threading

_state = {}
_boot_lock = threading.Lock()

_SESSION_MODULE = """module NoLimitsSession
using NoLimits
using NoLimits.Distributions
using NoLimits.OrdinaryDiffEq
using NoLimits.LinearAlgebra
import PythonCall
const DataFrames = NoLimits.DataFrames
nl_df(tbl) = DataFrames.DataFrame(tbl)
# Accessors such as get_random_effects return a NamedTuple of DataFrames, which
# DataFrame() would read as a one-row table of stringified vectors.
function nl_df(tbl::NamedTuple{<:Any, <:Tuple{DataFrames.DataFrame, Vararg{DataFrames.DataFrame}}})
    length(tbl) == 1 && return only(tbl)
    error("got a NamedTuple of $(length(tbl)) tables (fields: $(join(keys(tbl), ", "))); " *
          raw"collect one field, e.g. x.eta (Python) / x$eta (R)")
end
nl_pydf(x) = DataFrames.DataFrame(PythonCall.PyTable(x))
nl_pydf(x::DataFrames.DataFrame) = x  # already converted by the rule below
nl_names(x) = string.(Base.names(nl_df(x)))
nl_col(v) = eltype(v) <: Union{Real, AbstractString, Missing, Nothing} ? v : string.(v)
nl_table(x) = (df = nl_df(x); (; (Symbol(n) => nl_col(df[!, n]) for n in Base.names(df))...))
# `missing` crosses to Python as an opaque wrapper that pandas does not see as NA;
# `nothing` crosses as None, which pandas does. Numeric columns already arrive as
# NaN, so leave those alone and keep their float dtype.
# An all-missing column has eltype `Missing`, whose nonmissingtype is `Union{}` and
# therefore <: Real, so it needs its own clause or it would be treated as numeric.
nl_nona(v) = (eltype(v) === Missing ||
              (eltype(v) >: Missing && !(Base.nonmissingtype(eltype(v)) <: Real))) ?
    Any[x === missing ? nothing : x for x in v] : v
nl_pytable_df(x) = (df = nl_df(x);
    DataFrames.DataFrame([n => nl_nona(df[!, n]) for n in Base.names(df)]))
nl_eval_string(s::AbstractString) = Base.include_string(NoLimitsSession, s)
# Any pandas DataFrame passed to a Julia function converts to a DataFrame,
# so callers never have to convert by hand.
_pyconvert_pandas(::Type{DataFrames.DataFrame}, x) =
    PythonCall.pyconvert_return(nl_pydf(x))
for _t in ("pandas.core.frame:DataFrame", "pandas:DataFrame")
    # ARRAY priority outranks PythonCall's own PyPandasDataFrame rule.
    PythonCall.pyconvert_add_rule(
        _t, DataFrames.DataFrame, _pyconvert_pandas,
        PythonCall.PYCONVERT_PRIORITY_ARRAY,
    )
end
end"""


def _julia():
    """Boot Julia lazily on first use; memoize the session module.

    Locked: concurrent first calls would otherwise each define the session module,
    and the redefinition mid-call hangs the process. Julia must also boot on the
    main thread; booting it from a worker leaves a process that never exits, so
    that case raises instead of hanging.
    """
    if "session" not in _state:
        with _boot_lock:
            if "session" not in _state:
                if threading.current_thread() is not threading.main_thread():
                    raise RuntimeError(
                        "NoLimitsPy must boot Julia on the main thread. Call any "
                        "NoLimits function once from the main thread (e.g. "
                        "NoLimitsPy.seval('1+1')) before using NoLimitsPy from "
                        "worker threads; booting from a worker thread leaves a "
                        "process that never exits."
                    )
                from juliacall import Main  # first import resolves/installs the env
                Main.seval(_SESSION_MODULE)
                _state["NoLimits"] = Main.NoLimitsSession.NoLimits
                _state["session"] = Main.NoLimitsSession
    return _state["session"]


def __getattr__(name):
    # PEP 562: forward any unknown attribute to the NoLimits module.
    if name.startswith("_"):  # keep IPython/inspect probes from booting Julia
        raise AttributeError(name)
    _julia()
    try:
        return getattr(_state["NoLimits"], name)
    except AttributeError:
        raise AttributeError(f"NoLimits.jl has no binding {name!r}") from None


def __dir__():
    names = set(globals())
    if "session" in _state:
        names.update(str(s) for s in _julia().nl_eval_string("string.(names(NoLimits))"))
    return sorted(names)


def seval(code):
    """Evaluate Julia code (multiple expressions allowed) in the session module.

    Objects defined here (chains, helper functions) are visible to later model()
    calls, unlike juliacall's Main.seval.
    """
    return _julia().nl_eval_string(code)


def model(code):
    """Build a NoLimits Model from @Model DSL source.

    The string may contain a Julia prelude before @Model (e.g. `using SimpleChains`
    and chain definitions). A bare block body without @Model is wrapped in
    `@Model begin ... end`.
    """
    if not isinstance(code, str):
        raise TypeError(f"model() needs a str, got {type(code).__name__}")
    code = code.strip()
    # ponytail: a `#` inside a string literal also counts as a comment start here;
    # good enough to decide whether the source has a real @Model block.
    # A real @Model block opens a line, so a mid-line mention in a string literal
    # (or a comment) does not count.
    uncommented = (line.split("#", 1)[0] for line in code.splitlines())
    if not any(line.lstrip().startswith("@Model") for line in uncommented):
        code = "@Model begin\n" + code + "\nend"
    return seval(code)


def to_julia(df):
    """pandas.DataFrame (or any table) -> Julia DataFrames.DataFrame.

    Optional: pandas frames convert automatically when passed to any function.
    Use this to convert once and reuse the result across many calls.
    """
    return _julia().nl_pydf(df)


def collect(x):
    """Julia table (e.g. a predict() result) -> pandas.DataFrame."""
    jl = _julia()
    return jl.PythonCall.pytable(jl.nl_pytable_df(x))


def sym(s):
    """Julia Symbol from a string, for Symbol-typed arguments.

    Example: DataModel(model, jdf, primary_id=sym("ID"), time_col=sym("time")).
    """
    if not isinstance(s, str):
        raise TypeError(f"sym() needs a str, got {type(s).__name__}")
    if not s:
        raise ValueError("sym() needs a non-empty string")
    return _julia().Symbol(s)


def plot(fname, *args, file=None, **kwargs):
    """Call any NoLimits plot_* function by name; save to `file` (or a temp PNG).

    Returns the path. Displays inline (IPython) only when `file` is None.
    Loads CairoMakie and AlgebraOfGraphics on first use.
    """
    import os

    if "save_path" in kwargs:
        raise TypeError(
            "plot() sets save_path itself; pass the destination as file= "
            "instead of save_path= (not both)."
        )
    jl = _julia()
    if not _state.get("makie"):
        jl.nl_eval_string("import CairoMakie, AlgebraOfGraphics")
        _state["makie"] = True
    show = file is None
    if file is None:
        import tempfile
        fd, file = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    # NoLimits requires the save_path directory to exist; a bare filename has an
    # empty dirname, so hand Julia an absolute path.
    getattr(_state["NoLimits"], fname)(*args, save_path=os.path.abspath(file), **kwargs)
    # Makie leaves ~28 MB of figure memory behind per plot; a collect here recovers
    # part of it (~19 MB/plot residual, measured) and costs little next to rendering.
    jl.nl_eval_string("GC.gc(); nothing")
    if show:
        try:
            from IPython.display import Image, display
            display(Image(filename=file))
        except Exception:
            pass
    return file


def use_env(path):
    """Point juliapkg at a custom Julia project directory (creates/resolves it
    on first use). Must be called before the first NoLimits use; restart Python
    to switch afterwards. Default without this call: a Julia env inside the
    active virtualenv, so a new venv already gives a fresh isolated env."""
    if "session" in _state:
        raise RuntimeError(
            "Julia is already running; restart Python and call use_env "
            "before the first NoLimits use."
        )
    import os
    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(path)


def status():
    """Diagnostic report: versions, active Julia project, installed packages.

    Boots Julia lazily if needed."""
    import os
    info = {
        "started": "session" in _state,
        "target_project": os.environ.get("PYTHON_JULIAPKG_PROJECT"),
    }
    try:
        jl = _julia()
    except Exception as e:  # a boot failure is exactly what status() is for
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    info["started"] = True
    info["julia_version"] = str(jl.nl_eval_string("string(VERSION)"))
    info["project"] = str(jl.nl_eval_string("Base.active_project()"))
    info["nolimits_version"] = str(jl.nl_eval_string("string(pkgversion(NoLimits))"))
    info["installed_packages"] = sorted(
        str(s) for s in jl.nl_eval_string(
            "import Pkg; collect(keys(Pkg.project().dependencies))"
        )
    )
    return info


def update():
    """Pkg.update() the active Julia environment. If Julia is already loaded,
    updated versions take effect in the next Python session."""
    from juliacall import Main
    Main.seval("import Pkg")
    Main.Pkg.update()


def install_julia_packages(*pkgs):
    """Pkg.add extra Julia packages (e.g. "SimpleChains", "Lux", "CSV", "JLD2")."""
    from juliacall import Main
    # A Python list arrives as PyList{Any}, which Pkg.add has no method for, so
    # build the call in Julia source instead.
    spec = ", ".join('"{}"'.format(p) for p in pkgs)
    Main.seval("import Pkg; Pkg.add([{}])".format(spec))
