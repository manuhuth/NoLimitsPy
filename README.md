# NoLimitsPy

Python interface to [NoLimits.jl](https://github.com/manuhuth/NoLimits.jl), a Julia
package for nonlinear mixed-effects analysis of longitudinal data.

Every exported NoLimits.jl name is reachable as `NoLimitsPy.<name>`. The wrapper adds
only session management and a handful of conversion helpers, so new NoLimits features
are available as soon as the Julia package is updated.

## Installation

```bash
pip install NoLimitsPy
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

jdf = nl.to_julia(df)
dm = nl.DataModel(model, jdf, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
res = nl.fit_model(dm, nl.Laplace())

print(nl.get_objective(res))
print(nl.collect(nl.predict(res, jdf)).head())

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
  `time_col`. All other keyword arguments pass through as normal Python kwargs.
- Getting results back: `nl.collect(x)` turns a Julia table into a pandas DataFrame.
  Scalars convert automatically.
- `nl.seval(code)` evaluates Julia code, possibly several expressions, in the session
  module, so anything it defines is visible to later `nl.model` calls. Plain
  `juliacall.Main.seval` evaluates in `Main` instead and is not visible to model
  strings.
- Plotting: `nl.plot("plot_fits", res, file="fits.png")` calls any NoLimits `plot_*`
  function and forwards `save_path`. CairoMakie is loaded on the first plot call only.
- Model, DataModel, and fit-result objects are live references into the Julia session
  and do not survive the process. Persist fits with `save_fit`/`load_fit` after
  `nl.install_julia_packages("JLD2")`. A reloaded fit needs its model string
  re-evaluated, because models themselves are not serializable.

## License

MIT
