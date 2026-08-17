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
    # get_random_effects returns a NamedTuple of tables; collect() unwraps the
    # single field instead of stringifying it into one row.
    re = nl.collect(nl.get_random_effects(res))
    assert len(re) == 4
    assert sorted(re["ID"]) == ["s1", "s2", "s3", "s4"]
    assert re["eta_1"].dtype.kind == "f"
    out = tmp_path / "fits.png"
    nl.plot("plot_fits", res, file=str(out))
    assert out.stat().st_size > 0


def test_missing_values_round_trip_as_pandas_na():
    df = pd.DataFrame({"s": ["a", None, float("nan"), "b"]})
    back = nl.collect(nl.to_julia(df))
    assert back["s"].isna().tolist() == [False, True, True, False]
    assert back["s"].tolist()[0] == "a"


def test_all_missing_and_one_row_columns_come_back_as_na():
    # An all-missing column has Julia eltype `Missing`, which the first missing->None
    # fix skipped (nonmissingtype(Missing) is Union{}, a subtype of Real), so raw
    # Julia `missing` objects leaked into pandas.
    back = nl.collect(nl.to_julia(pd.DataFrame({"s": [None, None, None]})))
    assert back["s"].isna().tolist() == [True, True, True]
    # Same column type whatever the row count, and always NA-detectable. An all-NA
    # categorical reaches Julia as NaN (pandas sends the -1 codes as floats), so the
    # dtype is float64 rather than object; what matters is that it no longer depends
    # on the row count.
    cols = [nl.collect(nl.to_julia(pd.DataFrame(
        {"x": list(range(n)),
         "y": pd.Categorical([None] * n, categories=["a", "b"])})))["y"]
        for n in (1, 3)]
    assert cols[0].dtype == cols[1].dtype
    assert all(c.isna().all() for c in cols)


def test_model_rejects_at_model_inside_a_string_literal():
    import pytest
    # The heuristic only accepts @Model at the start of a line, so this body gets
    # wrapped and fails on the missing blocks instead of silently returning 2.
    with pytest.raises(Exception, match="@Model"):
        nl.model('msg = "call @Model to build a model"\nx = 1 + 1')


def test_sym_rejects_the_empty_string():
    import pytest
    with pytest.raises(ValueError):
        nl.sym("")


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


def test_model_rejects_non_strings():
    import pytest
    with pytest.raises(TypeError, match="needs a str"):
        nl.model(42)


def test_threads_after_a_main_thread_warmup_do_not_hang(tmp_path):
    # The lazy boot used to race when several threads made the first call at once
    # (double session module -> hung process). Booting from a worker thread hangs
    # the process at exit, so that path raises instead.
    import subprocess
    import sys
    script = tmp_path / "threads.py"
    script.write_text(
        "import threading\n"
        "import NoLimitsPy as nl\n"
        "cold = []\n"
        "def probe():\n"
        "    try:\n"
        "        nl.seval('1+1'); cold.append('no error')\n"
        "    except RuntimeError:\n"
        "        cold.append('raised')\n"
        "t = threading.Thread(target=probe); t.start(); t.join()\n"
        "print('COLD', cold[0])\n"
        "nl.seval('1+1')\n"
        "out = []\n"
        "lk = threading.Lock()\n"
        "def w(i):\n"
        "    for j in range(25):\n"
        "        v = int(nl.seval(f'{i} + {j}'))\n"
        "        with lk: out.append(v == i + j)\n"
        "ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]\n"
        "[t.start() for t in ts]\n"
        "[t.join() for t in ts]\n"
        "print('OK', len(out), all(out))\n"
    )
    out = subprocess.run([sys.executable, str(script)], env=dict(os.environ),
                         capture_output=True, text=True, timeout=600)
    assert "COLD raised" in out.stdout, out.stdout + out.stderr
    assert "OK 100 True" in out.stdout, out.stdout + out.stderr


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
