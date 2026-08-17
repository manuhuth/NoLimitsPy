# %% [markdown]
# # NoLimitsPy in VS Code
#
# Run this file cell by cell (Shift+Enter on a `# %%` cell opens the
# Interactive Window). The first cell boots Julia lazily; the very first run
# ever resolves the Julia environment and can take a few minutes, later runs
# start in seconds. Plots display inline in the Interactive Window.
#
# Requires NoLimits.jl >= 0.2.5 (until released: install the GitHub main
# revision).

# %% Setup
from pathlib import Path

import pandas as pd
import NoLimitsPy as nl

try:
    HERE = Path(__file__).parent
except NameError:  # Interactive Window / notebook cell
    HERE = Path.cwd()

# %% Model: one-compartment oral-absorption ODE, log-normal REs on ka, CL, V
MODEL = """
@fixedEffects begin
    ka       = RealNumber(1.5)
    cl       = RealNumber(0.04)
    v        = RealNumber(0.5)
    omega_ka = RealNumber(0.5, scale=:log)
    omega_cl = RealNumber(0.3, scale=:log)
    omega_v  = RealNumber(0.3, scale=:log)
    sigma    = RealNumber(0.7, scale=:log)
end

@covariates begin
    t    = Covariate()
    Dose = ConstantCovariate(constant_on=:Subject)
end

@randomEffects begin
    eta_ka = RandomEffect(LogNormal(0.0, omega_ka); column=:Subject)
    eta_cl = RandomEffect(LogNormal(0.0, omega_cl); column=:Subject)
    eta_v  = RandomEffect(LogNormal(0.0, omega_v);  column=:Subject)
end

@preDifferentialEquation begin
    kai = ka * eta_ka
    cli = cl * eta_cl
    vi  = v * eta_v
end

@DifferentialEquation begin
    D(depot)   ~ -kai * depot
    D(central) ~ kai * depot - (cli / vi) * central
end

@initialDE begin
    depot   = Dose
    central = 0.0
end

@formulas begin
    cp = central(t) / vi
    conc ~ Normal(cp, sigma)
end
"""
model = nl.model(MODEL)

# %% Data: theophylline (12 subjects), shipped next to the examples
df = pd.read_csv(HERE.parent / "theoph.csv")
df = df.rename(columns={"Time": "t"})
df["Subject"] = df["Subject"].astype(str)

dm = nl.DataModel(model, df, primary_id="Subject", time_col="t")

# %% Fit (warm-started Laplace)
res = nl.fit_model(dm, nl.Laplace(), pooled_init=True)
print("objective:      ", nl.get_objective(res))
print("marginal loglik:", nl.get_marginal_likelihood(res))
print("estimates:      ", nl.get_params(res, scale="untransformed"))

# %% Individual (EBE) predictions
pred = nl.collect(nl.predict(res, df, re_mode="ebe"))
pred.head(10)

# %% Plots (inline in the Interactive Window)
nl.plot("plot_fits", res)

# %%
nl.plot("plot_dv_pred", res)

# %%
