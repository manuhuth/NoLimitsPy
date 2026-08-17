# Theophylline population PK with NoLimitsPy.
#
# One-compartment oral-absorption ODE model (depot -> central), multiplicative
# log-normal random effects on ka, CL and V, fitted with warm-started Laplace.
# The ODE is linear, so NoLimits solves it in closed form.
#
# Run from the repository root in any Python where NoLimitsPy is installed
# (pip install -e . from the repository root, or
#  pip install "git+https://github.com/manuhuth/NoLimitsPy"):
#   python examples/theophylline.py
#
# Requires NoLimits.jl >= 0.2.5, which juliapkg resolves from the registry.

import pandas as pd
import NoLimitsPy as nl

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

# Theophylline data (12 subjects, oral dose in mg/kg, serum conc in mg/L),
# exported from R's datasets::Theoph.
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "theoph.csv")
df = df.rename(columns={"Time": "t"})
df["Subject"] = df["Subject"].astype(str)

# The pandas frame goes straight in; NoLimitsPy converts it on the way.
model = nl.model(MODEL)
dm = nl.DataModel(model, df, primary_id="Subject", time_col="t")

# Starting-value check before fitting: both must be finite, and :ebe > :mean.
theta0 = nl.get_params(dm, scale="untransformed")
print("loglik at theta0 (eta=mean):", nl.complete_data_loglikelihood(dm, theta0, eta="mean"))
print("loglik at theta0 (eta=ebe): ", nl.complete_data_loglikelihood(dm, theta0, eta="ebe"))

res = nl.fit_model(dm, nl.Laplace(), pooled_init=True)

print("objective:          ", nl.get_objective(res))
print("marginal loglik:    ", nl.get_marginal_likelihood(res))
print("estimates (natural):", nl.get_params(res, scale="untransformed"))
print("shrinkage:          ", nl.compute_shrinkage(res))

# Individual (EBE) predictions on the observed grid.
pred = nl.collect(nl.predict(res, df, re_mode="ebe"))
print(pred.head(10))

# Diagnostics, saved as PNG.
nl.plot("plot_fits", res, file="theoph_fits.png")
nl.plot("plot_dv_pred", res, file="theoph_dv_pred.png")
print("plots written: theoph_fits.png, theoph_dv_pred.png")
