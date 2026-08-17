# NoLimitsPy

[![CI](https://github.com/manuhuth/NoLimitsPy/actions/workflows/ci.yml/badge.svg)](https://github.com/manuhuth/NoLimitsPy/actions/workflows/ci.yml)

Python interface to [NoLimits.jl](https://github.com/manuhuth/NoLimits.jl), a Julia
package for nonlinear mixed-effects analysis of longitudinal data.

Every exported NoLimits.jl name is reachable as `NoLimitsPy.<name>`. The wrapper adds
only session management and a handful of conversion helpers, so new NoLimits features
are available as soon as the Julia package is updated.

## Installation

```bash
pip install "git+https://github.com/manuhuth/NoLimitsPy"
```

Julia and the NoLimits.jl packages are resolved automatically on first import by
[juliapkg](https://github.com/JuliaPy/pyjuliapkg). The first import therefore
downloads Julia if needed and precompiles NoLimits.jl, which takes several minutes.
Later imports are fast.

`NoLimitsPy.collect` returns a pandas DataFrame, so install `pandas` if you want to
pull tables back into Python.

## Environments

By default juliapkg puts the Julia environment inside the active virtualenv, so a
new venv already gives a fresh, isolated set of Julia packages. To place it
somewhere else, point juliapkg at a project directory before the first NoLimits
use:

```python
import NoLimitsPy as nl

nl.use_env("julia-envs/myproject")   # created and resolved on first use
```

`nl.status()` is the first thing to run when something looks wrong. It reports the
Julia and NoLimits.jl versions, the active Julia project, and the packages
installed in it.

`nl.update()` runs `Pkg.update()` on the active environment. If Julia is already
loaded, the updated versions take effect in the next Python session, because
loaded Julia code is never replaced in place.

The environment is a boot-time choice for the same reason: Julia cannot unload a
package. Call `use_env` before the first NoLimits use, and restart the Python
interpreter to switch afterwards.

## Quickstart

```python
import pandas as pd
import NoLimitsPy as nl

model = nl.model("""
@fixedEffects begin
    A0    = RealNumber(10.0, scale=:log)
    k     = RealNumber(0.5,  scale=:log)
    omega = RealNumber(0.3,  scale=:log)
    sigma = RealNumber(0.5,  scale=:log)
end

@covariates begin
    time = Covariate()
end

@randomEffects begin
    eta = RandomEffect(Normal(0.0, omega); column=:ID)
end

@formulas begin
    pred = A0 * exp(eta) * exp(-k * time)
    y ~ Normal(pred, sigma)
end
""")

df = pd.DataFrame({
    "ID": [s for s in ["s1", "s2", "s3", "s4"] for _ in range(4)],
    "time": [0.0, 1.0, 2.0, 4.0] * 4,
    "y": [10.2, 6.1, 3.6, 1.4, 12.5, 7.8, 4.9, 1.9,
          8.1, 4.9, 3.0, 1.1, 11.0, 6.5, 4.1, 1.6],
})

dm = nl.DataModel(model, df, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
res = nl.fit_model(dm, nl.Laplace())

print(nl.get_objective(res))
print(nl.collect(nl.predict(res, df)).head())

nl.plot("plot_fits", res, file="fits.png")
```

## Models with a Julia prelude (neural networks and other extras)

`nl.model` accepts arbitrary Julia code before the `@Model` block, which is how
neural-network parameters, custom functions, and weak-dependency imports enter a
model. Install the optional Julia package once:

```python
nl.install_julia_packages("SimpleChains")
```

Then put the prelude in the model string:

```python
model = nl.model("""
using SimpleChains
chain = SimpleChain(static(1), TurboDense(tanh, 2), TurboDense(identity, 1))

@Model begin
    @fixedEffects begin
        z = NNParameters(chain; function_name=:NN1, calculate_se=false)
        sigma = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        time = Covariate()
    end
    @formulas begin
        pred = NN1([time], z)[1]
        y ~ Normal(pred, sigma)
    end
end
""")
```

The same mechanism covers Lux, CSV, JLD2, Copulas, and Turing.

## Notes

- Dynamic surface: any NoLimits.jl export is available as `NoLimitsPy.<name>`. See the
  NoLimits.jl documentation for the API, the wrapper adds nothing on top of it.
- Symbols: use `nl.sym("ID")` for Symbol-typed arguments such as `primary_id` and
  `time_col`. The name must be a non-empty string. All other keyword arguments pass through as normal Python kwargs.
- Passing data in: any function accepts a pandas DataFrame directly, it is converted
  to a Julia DataFrame at the call boundary. `nl.to_julia(df)` converts once up front,
  which is worth doing when the same frame goes into many calls.
- Getting results back: `nl.collect(x)` turns a Julia table into a pandas DataFrame.
  Scalars convert automatically. Julia `missing` comes back as `None`, which pandas
  treats as NA, so `isna()` and `dropna()` work on the result.
- Conversion caveats, all of them consequences of what a Julia `DataFrame` can hold:
  - A pandas index is dropped. Julia DataFrames have no index concept, so a custom
    index or a MultiIndex is lost silently and the round trip returns a plain
    `RangeIndex`. Call `df.reset_index()` first when those levels are data.
  - Duplicate column names are renamed. DataFrames.jl requires unique names, so
    `["x", "x"]` becomes `["x", "x_2"]` without a warning. Rename before converting.
  - An all-NA column comes back NA-detectable but not always in its original dtype: an
    all-NA `Categorical` reaches Julia as `NaN` (pandas sends the -1 codes as floats),
    so it returns as `float64` rather than `object`. The dtype no longer depends on the
    row count.
  - Nullable extension dtypes collapse. `Int64`, `boolean`, and `Categorical` columns
    come back as `object` or `string` columns; values survive, the declared dtype and
    a Categorical's category order do not.
- Dict options: a dict whose keys are all strings or symbols becomes a Julia
  NamedTuple, which is what options such as `optim_kwargs={"show_trace": True}`
  expect. A dict keyed by anything else stays a Julia dictionary and is passed
  through untouched, which is deliberate: NoLimits options such as `constants_re`
  are keyed by values rather than by names.
- Model strings: a `@Model` block is recognized when `@Model` opens a line, so a
  mention of it inside a comment or a string literal does not stop the body from being
  wrapped in `@Model begin ... end`.
- `nl.seval(code)` evaluates Julia code, possibly several expressions, in the session
  module, so anything it defines is visible to later `nl.model` calls. Evaluating in
  Julia's `Main` instead, which is what the underlying bridge does by default, is not
  visible to model strings.
- Julia console output, for example optimizer traces from `optim_kwargs` with
  `show_trace` or progress logs, streams to the Python console automatically. In
  Jupyter or the VS Code Interactive Window, raw Julia output may appear in the
  terminal running the kernel instead of in the cell.
- Wide tables are the expensive case for `nl.collect()`: its cost tracks the number of
  columns, not the number of rows, at roughly 3 ms per column. Measured on one machine,
  500,000 rows by 5 columns collects in about 0.6 s, while 50 rows by 1000 columns takes
  about 2.8 s and 50 by 2000 about 5.5 s. Reshape a wide result table (one column per
  parameter, subject, or simulation draw) to long form in Julia before collecting it.
- Plotting: `nl.plot("plot_fits", res, file="fits.png")` calls any NoLimits `plot_*`
  function and forwards `save_path`. CairoMakie is loaded on the first plot call only.
- Plot memory: Makie leaves figure memory behind on every plot call, about 28 MB per
  plot. Because juliacall embeds Julia in the Python process, this shows up as growing
  RSS. `nl.plot()` runs a Julia garbage collection after saving, which recovers part of
  it (about 19 MB per plot remained in a measured 10-plot loop), so a long plotting loop
  still grows; run such loops in a subprocess if memory matters.
- Threads: the first NoLimits call boots Julia, and Julia must boot on the main thread.
  Call anything once from the main thread (`nl.seval("1+1")` will do) before using
  NoLimitsPy from worker threads; a boot triggered from a worker raises rather than
  leaving a process that never exits.
- Object lifetime: Model, DataModel, and fit-result objects are live references into the
  Julia session and do not survive the process. `pickle.dump` on one appears to succeed
  and even round-trips through `pickle.loads` in the same process, but loading the file
  in a new process fails with a bare `Exception: error deserializing this value`;
  `copy.deepcopy` is unreliable for the same reason. Persist fits with
  `save_fit`/`load_fit` after `nl.install_julia_packages("JLD2")`. A reloaded fit needs
  its model string re-evaluated, because models themselves are not serializable. For a
  readable record of an object, save `str(x)`, which is Julia's own printout. Objects
  such as `get_random_effects(res)` that hold one table per random effect are collected
  per field, as in `nl.collect(re.eta)`; `nl.collect(re)` unwraps a single field for you
  and errors, listing the field names, when there are several.

## Examples

- `examples/theophylline.py` is a runnable population-PK script: a
  one-compartment oral-absorption ODE model on the Theophylline data, fitted with
  warm-started Laplace, with fit and diagnostic plots written to PNG.

  ```bash
  python examples/theophylline.py
  ```

- `examples/vscode-python/` is a ready-to-open VS Code project with the same
  model as a cell-by-cell script that renders plots inline. See its
  [README](examples/vscode-python/README.md).

## Related

- [NoLimits.jl](https://github.com/manuhuth/NoLimits.jl), the Julia package this
  wrapper exposes.
- [NoLimitsR](https://github.com/manuhuth/NoLimitsR), the sibling R interface.

## License

MIT, see [LICENSE](LICENSE).
