import math

import pandas as pd
import NoLimitsPy as nl

MODEL = """
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
"""


def test_seval():
    assert nl.seval("1 + 1") == 2


def test_roundtrip(tmp_path):
    model = nl.model(MODEL)
    df = pd.DataFrame({
        "ID": [s for s in ["s1", "s2", "s3", "s4"] for _ in range(4)],
        "time": [0.0, 1.0, 2.0, 4.0] * 4,
        "y": [10.2, 6.1, 3.6, 1.4, 12.5, 7.8, 4.9, 1.9,
              8.1, 4.9, 3.0, 1.1, 11.0, 6.5, 4.1, 1.6],
    })
    jdf = nl.to_julia(df)
    dm = nl.DataModel(model, jdf, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
    res = nl.fit_model(dm, nl.Laplace())
    assert math.isfinite(float(nl.get_objective(res)))
    pred = nl.predict(res, jdf)
    assert "prediction" in list(nl.collect(pred).columns)
    out = tmp_path / "fits.png"
    nl.plot("plot_fits", res, file=str(out))
    assert out.stat().st_size > 0


def test_status_and_use_env_guard():
    info = nl.status()
    assert info["nolimits_version"]
    assert "NoLimits" in info["installed_packages"]
    import pytest
    with pytest.raises(RuntimeError):
        nl.use_env("/some/other/env")
