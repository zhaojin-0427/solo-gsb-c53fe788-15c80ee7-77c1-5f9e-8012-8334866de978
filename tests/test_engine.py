"""数值正确性单元测试：抽样分布、GUM 解析解、MC 大数收敛。"""

from __future__ import annotations

import math

import numpy as np

from app.engine import _marginal_unit_samples, _phi, _norm_ppf


def test_phi_matches_known_values():
    assert abs(float(_phi(np.array([0.0]))[0]) - 0.5) < 1e-7
    assert abs(float(_phi(np.array([1.96]))[0]) - 0.975) < 1e-5
    assert abs(float(_phi(np.array([-1.96]))[0]) - 0.025) < 1e-5
    assert abs(float(_phi(np.array([3.0]))[0]) - 0.99865) < 1e-5


def test_norm_ppf_inverts_phi():
    p = np.linspace(0.001, 0.999, 999)
    z = _norm_ppf(p)
    back = _phi(z)
    assert np.max(np.abs(back - p)) < 3e-6
    assert abs(float(z[np.argmin(np.abs(p - 0.975))]) - 1.96) < 1e-4


def test_marginal_distributions_have_unit_variance():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=500_000)
    for dist, target_mean in (("normal", 0.0), ("uniform", 0.0), ("triangular", 0.0)):
        s = _marginal_unit_samples(dist, u.copy())
        assert abs(s.mean() - target_mean) < 2e-3
        assert abs(s.std(ddof=1) - 1.0) < 2e-3


def test_triangular_support_bounds():
    u = np.linspace(1e-9, 1 - 1e-9, 10000)
    s = _marginal_unit_samples("triangular", u)
    assert s.min() >= -math.sqrt(6.0) - 1e-6
    assert s.max() <= math.sqrt(6.0) + 1e-6


def test_gum_linear_model_exact_variance():
    # y = 2*a + 3*b（无量纲线性），相关 r -> GUM 必须与解析解一致
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_gum

    expr = compile_expression("2*a + 3*b", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = ctx.nominal_to_base({"a": 1.0, "b": -2.0})
    std = np.array([0.1, 0.2])
    for r in (-0.7, 0.0, 0.5):
        corr = build_correlation_matrix(("a", "b"), [("a", "b", r)], set())
        g = run_gum(expr, ("a", "b"), x, std, corr, ctx)
        expected_var = (2 * 0.1) ** 2 + (3 * 0.2) ** 2 + 2 * 2 * 3 * r * 0.1 * 0.2
        assert abs(g.y - (-4.0)) < 1e-9
        assert abs(g.uc - math.sqrt(expected_var)) < 1e-9


def test_monte_carlo_matches_gum_for_linear_gaussian():
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_monte_carlo

    expr = compile_expression("2*a + 3*b", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = {"a": 0.0, "b": 0.0}
    std = np.array([0.1, 0.2])
    corr = build_correlation_matrix(("a", "b"), [("a", "b", 0.5)], set())
    mc = run_monte_carlo(
        expr, ("a", "b"), ("normal", "normal"), x, std, corr,
        n_draws=100_000, seed=123, ctx=ctx, convergence_tol=0.1,
    )
    expected_sd = math.sqrt(0.2 ** 2 + 0.6 ** 2 + 2 * 0.5 * 0.2 * 0.6)
    assert abs(mc.mean) < 3e-3
    assert abs(mc.std - expected_sd) < 3e-3
    lo, hi = -1.96 * expected_sd, 1.96 * expected_sd
    assert abs(mc.coverage_low - lo) < 1.5e-2
    assert abs(mc.coverage_high - hi) < 1.5e-2
    assert mc.converged


def test_reproducibility_fixed_seed():
    from app.expressions import compile_expression
    from app.units import build_unit_context, parse_unit
    from app.engine import build_correlation_matrix, run_monte_carlo

    expr = compile_expression("a*b + sqrt(a**2)", ("a", "b"))
    units = {"a": parse_unit("dimensionless"), "b": parse_unit("dimensionless")}
    ctx = build_unit_context(expr, units, "dimensionless")
    x = {"a": 2.0, "b": 3.0}
    std = np.array([0.05, 0.05])
    corr = np.eye(2)
    kw = dict(compiled=expr, names=("a", "b"), distributions=("normal", "normal"),
              x_base=x, std_base=std, corr=corr, n_draws=5000, ctx=ctx, convergence_tol=0.5)
    m1 = run_monte_carlo(seed=99, **kw)
    m2 = run_monte_carlo(seed=99, **kw)
    assert m1.mean == m2.mean and m1.coverage_low == m2.coverage_low
    m3 = run_monte_carlo(seed=100, **kw)
    assert m3.mean != m1.mean


def test_psd_repair_accepts_tiny_negative_eigenvalue():
    from app.engine import build_correlation_matrix, CorrelationError
    # 合法相关系数（全部 <= 1）组成、但整体含 -5e-10 量级负特征值的近奇异矩阵
    pairs = [
        ("a", "b", 1.0 - 1e-11),
        ("a", "c", 1.0 - 1e-11),
        ("b", "c", 0.9999999995),
    ]
    m = np.eye(3)
    m[0, 1] = m[1, 0] = pairs[0][2]
    m[0, 2] = m[2, 0] = pairs[1][2]
    m[1, 2] = m[2, 1] = pairs[2][2]
    assert -1e-8 < np.linalg.eigvalsh(m)[0] < 0
    r = build_correlation_matrix(("a", "b", "c"), pairs, set())
    assert r.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(r) >= -1e-12)
    # 明显非 PSD 拒绝
    try:
        build_correlation_matrix(("a", "b", "c"),
                                 [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", -0.9)], set())
        assert False, "应当拒绝非半正定矩阵"
    except CorrelationError:
        pass
