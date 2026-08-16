"""Python interface to NoLimits.jl.

Every exported NoLimits.jl name is available as ``NoLimitsPy.<name>`` via dynamic
forwarding; only the helpers defined here are hand-written. New NoLimits features
appear automatically with package updates - by design, do not add named wrappers.
"""

_state = {}

_SESSION_MODULE = """module NoLimitsSession
using NoLimits
using NoLimits.Distributions
using NoLimits.OrdinaryDiffEq
using NoLimits.LinearAlgebra
import PythonCall
const DataFrames = NoLimits.DataFrames
nl_df(tbl) = DataFrames.DataFrame(tbl)
nl_pydf(x) = DataFrames.DataFrame(PythonCall.PyTable(x))
nl_names(x) = string.(Base.names(nl_df(x)))
nl_table(x) = (df = nl_df(x); (; (Symbol(n) => df[!, n] for n in Base.names(df))...))
nl_eval_string(s::AbstractString) = Base.include_string(NoLimitsSession, s)
end"""


def _julia():
    """Boot Julia lazily on first use; memoize the session module."""
    if "session" not in _state:
        from juliacall import Main  # first import resolves/installs the Julia env
        Main.seval(_SESSION_MODULE)
        _state["session"] = Main.NoLimitsSession
        _state["NoLimits"] = Main.NoLimitsSession.NoLimits
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
    code = code.strip()
    if "@Model" not in code:
        code = "@Model begin\n" + code + "\nend"
    return seval(code)


def to_julia(df):
    """pandas.DataFrame (or any table) -> Julia DataFrames.DataFrame."""
    return _julia().nl_pydf(df)


def collect(x):
    """Julia table (e.g. a predict() result) -> pandas.DataFrame."""
    jl = _julia()
    return jl.PythonCall.pytable(jl.nl_df(x))


def sym(s):
    """Julia Symbol from a string, for Symbol-typed arguments.

    Example: DataModel(model, jdf, primary_id=sym("ID"), time_col=sym("time")).
    """
    return _julia().Symbol(s)


def plot(fname, *args, file=None, **kwargs):
    """Call any NoLimits plot_* function by name; save to `file` (or a temp PNG).

    Returns the path. Displays inline (IPython) only when `file` is None.
    Loads CairoMakie on first use.
    """
    jl = _julia()
    if not _state.get("makie"):
        jl.nl_eval_string("import CairoMakie")
        _state["makie"] = True
    show = file is None
    if file is None:
        import os
        import tempfile
        fd, file = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    getattr(_state["NoLimits"], fname)(*args, save_path=str(file), **kwargs)
    if show:
        try:
            from IPython.display import Image, display
            display(Image(filename=file))
        except Exception:
            pass
    return file


def install_julia_packages(*pkgs):
    """Pkg.add extra Julia packages (e.g. "SimpleChains", "Lux", "CSV", "JLD2")."""
    from juliacall import Main
    # A Python list arrives as PyList{Any}, which Pkg.add has no method for, so
    # build the call in Julia source instead.
    spec = ", ".join('"{}"'.format(p) for p in pkgs)
    Main.seval("import Pkg; Pkg.add([{}])".format(spec))
