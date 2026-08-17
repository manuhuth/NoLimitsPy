import math
import os

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
    # Raw pandas frame: the pyconvert rule converts it at the call boundary.
    dm = nl.DataModel(model, df, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
    res = nl.fit_model(dm, nl.Laplace())
    assert math.isfinite(float(nl.get_objective(res)))
    pred = nl.predict(res, nl.to_julia(df))
    assert "prediction" in list(nl.collect(pred).columns)
    out = tmp_path / "fits.png"
    nl.plot("plot_fits", res, file=str(out))
    assert out.stat().st_size > 0


def test_missing_values_round_trip_as_pandas_na():
    df = pd.DataFrame({"s": ["a", None, float("nan"), "b"]})
    back = nl.collect(nl.to_julia(df))
    assert back["s"].isna().tolist() == [False, True, True, False]
    assert back["s"].tolist()[0] == "a"


def test_model_wraps_bare_body_with_at_model_in_a_comment():
    m = nl.model("""
# no @Model here, just a comment
@fixedEffects begin
    sigma = RealNumber(0.5, scale=:log)
end
@covariates begin
    time = Covariate()
end
@formulas begin
    y ~ Normal(time, sigma)
end
""")
    assert "Model" in str(nl.seval("typeof")(m))


def test_sym_rejects_non_strings():
    import pytest
    with pytest.raises(TypeError):
        nl.sym(["ID"])


def test_plot_rejects_save_path_alongside_file():
    import pytest
    with pytest.raises(TypeError, match="save_path"):
        nl.plot("plot_fits", None, file="a.png", save_path="b.png")


def test_status_reports_a_boot_failure_instead_of_raising(tmp_path):
    # A file where juliapkg expects a project directory makes the boot fail.
    import subprocess
    import sys
    bad = tmp_path / "not_a_dir"
    bad.write_text("")
    code = (
        "import os, NoLimitsPy as nl;"
        "info = nl.status();"
        "print('ERRKEY', 'error' in info)"
    )
    env = dict(os.environ, PYTHON_JULIAPKG_PROJECT=str(bad),
               PYTHON_JULIAPKG_OFFLINE="yes")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert "ERRKEY True" in out.stdout, out.stdout + out.stderr


def test_status_and_use_env_guard():
    info = nl.status()
    assert info["nolimits_version"]
    assert "NoLimits" in info["installed_packages"]
    import pytest
    with pytest.raises(RuntimeError):
        nl.use_env("/some/other/env")
