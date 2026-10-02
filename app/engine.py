"""不确定度传播计算引擎：GUM 一阶线性化 + 蒙特卡洛。

所有数值计算在调用方给定的 *基本单位量纲空间* 中完成（见 :mod:`app.units`），
引擎本身不感知单位，只感知数值、标准差与相关系数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .errors import CorrelationError, PropagationError, ConvergenceError
from .expressions import CompiledExpression, numpy_evaluation

# ---------------------------------------------------------------------------
# 分布
# ---------------------------------------------------------------------------

#: 支持的输入量分布
DISTRIBUTIONS = ("normal", "uniform", "triangular")

_EPS = np.finfo(float).eps
_EPS13 = _EPS ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 相关矩阵
# ---------------------------------------------------------------------------

def build_correlation_matrix(
    names: tuple[str, ...],
    pairs: list[tuple[str, str, float]],
    zero_std: set[str],
) -> np.ndarray:
    """由 (i, j, r) 对构造并校验相关系数矩阵。

    * 对称、对角为 1、|r| <= 1；
    * 不允许引用标准不确定度为 0 的输入量（相关系数在那里无定义）；
    * 必须半正定，但**允许奇异**（零特征值）：r = ±1 的完全相关是合法物理情形，
      抽样阶段用特征分解因子处理（见 :func:`_correlation_factor`）；
    * 最小特征值落在 [-1e-8, 0) 的浮点舍入误差投影修复到半正定（保持单位对角），
      低于 -1e-8 的明显非半正定矩阵拒绝。
    """
    n = len(names)
    idx = {name: k for k, name in enumerate(names)}
    rmat = np.eye(n)

    for a, b, r in pairs:
        if a == b:
            raise CorrelationError(f"输入量 {a!r} 不能与自身指定相关系数")
        if a in zero_std or b in zero_std:
            raise CorrelationError(
                f"相关系数 ({a!r}, {b!r}) 引用了标准不确定度为 0 的输入量"
            )
        if not -1.0 <= r <= 1.0:
            # 允许浮点序列化造成的 1±1e-12 级别越界，明显越界仍拒绝
            if abs(r) <= 1.0 + 1e-12:
                r = max(-1.0, min(1.0, r))
            else:
                raise CorrelationError(f"相关系数 ({a!r}, {b!r}) 越界: r={r}")
        ia, ib = idx[a], idx[b]
        if rmat[ia, ib] != 0.0:
            raise CorrelationError(f"相关系数 ({a!r}, {b!r}) 重复指定")
        rmat[ia, ib] = r
        rmat[ib, ia] = r

    if n == 0:
        return rmat

    eig = np.linalg.eigvalsh(rmat)
    # 允许 1e-8 绝对量级 + 与最大特征值成正比的浮点误差
    tol = max(1e-8, n * _EPS * eig[-1])
    if eig[0] < -tol:
        worst = eig[0]
        raise CorrelationError(
            f"相关矩阵不是半正定矩阵：最小特征值 {worst:.3e} < -{tol:.1e}"
        )

    # 浮点钳位：用 Higham 交替投影把矩阵投影到相关矩阵锥（保对角 1 + 半正定）。
    # 注意：完全相关（r=±1）的矩阵本来就是奇异的，零特征值原样保留，
    # 不再要求 Cholesky 可分解。
    if eig[0] < 0:
        rmat = _nearest_correlation_matrix(rmat)
        if np.linalg.eigvalsh(rmat)[0] < -1e-10:  # pragma: no cover - 防御式
            raise CorrelationError("相关矩阵投影修复后仍不是半正定矩阵")

    return rmat


def _nearest_correlation_matrix(a: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Dykstra 交替投影求最近相关矩阵（Higham 2002 简化版）。

    收敛判据只看半正定与单位对角；奇异（零特征值）结果合法——
    完全相关的目标矩阵本身就是奇异的，不要求 Cholesky 可分解。
    """
    y = a.copy()
    delta = np.zeros_like(a)
    for _ in range(max_iter):
        # 投影 1：谱半正定投影
        evals, evecs = np.linalg.eigh(y - delta)
        x = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
        # Dykstra 修正量
        delta = x - (y - delta)
        # 投影 2：对角归一化（单位对角）
        d = np.sqrt(np.diag(x))
        d[d == 0] = 1.0
        y = x / np.outer(d, d)
        np.fill_diagonal(y, 1.0)
        if np.min(np.linalg.eigvalsh(y)) >= -1e-12:
            break
    return y


def _correlation_factor(corr: np.ndarray) -> np.ndarray:
    """返回满足 ``L @ L.T == corr`` 的方阵因子（用于相关正态抽样 Z = L W）。

    非奇异矩阵用 Cholesky 分解（与既有行为位级一致，保证旧请求结果不变）；
    奇异半正定矩阵（如 r=±1 完全相关）退化为特征分解平方根
    ``L = Q·sqrt(Λ)``：零特征值方向自然消失，完全相关的输入由同一组
    标准正态变量驱动——r=1 时两路标准化样本同向、r=-1 时反向。
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(corr)
        evals = np.clip(evals, 0.0, None)  # 钳掉 -1e-16 量级的浮点残差
        return evecs * np.sqrt(evals)


# ---------------------------------------------------------------------------
# 边际分布抽样（Gaussian copula，相关结构正确且边际分布精确）
# ---------------------------------------------------------------------------

def _phi(z: np.ndarray) -> np.ndarray:
    """标准正态 CDF（Abramowitz & Stegun 26.2.17，绝对误差 < 7.5e-8）。"""
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * az)
    poly = (((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t)
    tail = poly * np.exp(-0.5 * az * az) / math.sqrt(2.0 * math.pi)
    return np.where(z >= 0.0, 1.0 - tail, tail)


def _marginal_unit_samples(distribution: str, u: np.ndarray) -> np.ndarray:
    """把均匀概率 u 映射为 *标准差为 1* 的目标边际分布样本。"""
    if distribution == "normal":
        # 标准正态分位数：Acklam 近似（绝对误差 ~1.15e-9）
        return _norm_ppf(u)
    if distribution == "uniform":
        # U(-sqrt(3), sqrt(3))：标准差为 1 的均匀分布
        return math.sqrt(3.0) * (2.0 * u - 1.0)
    if distribution == "triangular":
        # 对称三角(-1,0,1)，标准差 1/sqrt(6)，缩放后标准差为 1
        out = np.empty_like(u)
        lo = u <= 0.5
        out[lo] = -1.0 + np.sqrt(2.0 * u[lo])
        out[~lo] = 1.0 - np.sqrt(2.0 * (1.0 - u[~lo]))
        return math.sqrt(6.0) * out
    raise PropagationError(f"不支持的分布: {distribution}")  # pragma: no cover


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """标准正态逆 CDF（Peter Acklam 近似）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)

    m1 = p < plow
    q = np.sqrt(-2.0 * np.log(p[m1]))
    out[m1] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    m3 = p > phigh
    q = np.sqrt(-2.0 * np.log(1.0 - p[m3]))
    out[m3] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    m2 = ~(m1 | m3)
    q = p[m2] - 0.5
    r = q * q
    out[m2] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
              (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


# ---------------------------------------------------------------------------
# 数值微分
# ---------------------------------------------------------------------------

def _evaluate_scalar(compiled: CompiledExpression, env_base: Mapping[str, float]) -> float:
    y = numpy_evaluation(compiled, env_base)
    yv = float(np.asarray(y, dtype=float))
    if not math.isfinite(yv):
        raise PropagationError("模型在名义点（或其邻域）产生了非有限结果（NaN/Inf），无法传播不确定度")
    return yv


def numerical_gradient(compiled: CompiledExpression, x_base: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    """中心差分灵敏度 dy/dx（基本单位空间）。"""
    grad = np.zeros(len(names))
    for k, name in enumerate(names):
        x0 = x_base[name]
        h = _EPS13 * max(abs(x0), _EPS13)
        env_plus = dict(x_base)
        env_minus = dict(x_base)
        env_plus[name] = x0 + h
        env_minus[name] = x0 - h
        yp = _evaluate_scalar(compiled, env_plus)
        ym = _evaluate_scalar(compiled, env_minus)
        grad[k] = (yp - ym) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# GUM 一阶
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GumResult:
    y: float
    uc: float
    coverage_low: float
    coverage_high: float
    k: float
    sensitivities: dict[str, float]
    contributions: list[dict[str, float | str]]


def run_gum(
    compiled: CompiledExpression,
    names: tuple[str, ...],
    x_base: dict[str, float],
    std_base: np.ndarray,
    corr: np.ndarray,
    ctx,
) -> GumResult:
    """GUM 一阶线性化（输出空间）。"""
    y_base = _evaluate_scalar(compiled, x_base)
    grad_base = numerical_gradient(compiled, x_base, names)
    cov = (std_base[:, None] * corr * std_base[None, :])
    var_y = float(grad_base @ cov @ grad_base)
    if var_y < 0 and var_y > -_EPS * max(1.0, abs(grad_base @ (np.diag(np.diag(cov))) @ grad_base)):
        var_y = 0.0
    if var_y < 0:
        raise PropagationError(f"GUM 合成方差为负 ({var_y:.3e})")
    uc_base = math.sqrt(var_y)

    y = float(ctx.from_base(y_base))
    uc = ctx.output_scale * uc_base
    coverage_factor = 1.959963984540054  # 95% 正态覆盖因子
    sensitivities = {name: ctx.sensitivity(name, float(grad_base[ki])) for ki, name in enumerate(names)}

    # 方差贡献分解：对角项 + 协方差对
    contributions: list[dict[str, float | str]] = []
    total_var = var_y if var_y > 0 else 1.0
    for ki, name in enumerate(names):
        v = grad_base[ki] ** 2 * cov[ki, ki]
        contributions.append({
            "input": name,
            "variance_contribution": (ctx.output_scale ** 2) * v,
            "fraction": v / total_var if var_y > 0 else 0.0,
            "kind": "individual",
        })
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = 2.0 * grad_base[i] * grad_base[j] * cov[i, j]
            if v != 0.0:
                contributions.append({
                    "input": f"{names[i]}~{names[j]}",
                    "variance_contribution": (ctx.output_scale ** 2) * v,
                    "fraction": v / total_var if var_y > 0 else 0.0,
                    "kind": "covariance",
                })

    return GumResult(
        y=y,
        uc=uc,
        coverage_low=y - coverage_factor * uc,
        coverage_high=y + coverage_factor * uc,
        k=coverage_factor,
        sensitivities=sensitivities,
        contributions=contributions,
    )


# ---------------------------------------------------------------------------
# 蒙特卡洛
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class McResult:
    mean: float
    std: float
    median: float
    coverage_low: float
    coverage_high: float
    samples_min: float
    samples_max: float
    input_correlations: list[dict[str, float]]
    convergence: dict[str, float]
    converged: bool
    nonconvergence_reasons: list[str]


def _gaussian_density(x: float, samples: np.ndarray, sd: float) -> float:
    """Silverman 高斯核密度估计（用于分位标准误）。"""
    n = samples.size
    h = 1.06 * sd * n ** (-0.2) if sd > 0 else 1.0
    if h == 0:
        return 0.0
    z = (x - samples) / h
    return float(np.exp(-0.5 * z * z).sum() / (n * h * math.sqrt(2.0 * math.pi)))


def run_monte_carlo(
    compiled: CompiledExpression,
    names: tuple[str, ...],
    distributions: tuple[str, ...],
    x_base: dict[str, float],
    std_base: np.ndarray,
    corr: np.ndarray,
    n_draws: int,
    seed: int,
    ctx,
    convergence_tol: float,
) -> McResult:
    """N 次蒙特卡洛传播 + 收敛统计。"""
    rng = np.random.default_rng(seed)
    n = len(names)
    means = np.array([x_base[name] for name in names], dtype=float)

    # 相关标准正态：Z = L W；L 支持奇异相关矩阵（完全相关时零特征值方向消失）
    lower = _correlation_factor(corr)
    w = rng.standard_normal(size=(n, n_draws))
    z = lower @ w
    u = _phi(z)
    u = np.clip(u, 1e-12, 1 - 1e-12)  # 数值保护

    draws = np.empty((n, n_draws))
    for k, dist in enumerate(distributions):
        draws[k] = means[k] + std_base[k] * _marginal_unit_samples(dist, u[k])

    env = {name: draws[k] for k, name in enumerate(names)}
    # 抽样可能越过模型定义域（如 sqrt/log），numpy 会给出 NaN + RuntimeWarning；
    # 下面立即统计非有限值并明确拒绝，告警本身无意义，压掉避免污染日志。
    with np.errstate(all="ignore"):
        y_base = numpy_evaluation(compiled, env)
    y_base = np.asarray(y_base, dtype=float)

    if np.any(~np.isfinite(y_base)):
        bad = int(np.sum(~np.isfinite(y_base)))
        raise PropagationError(
            f"蒙特卡洛抽样中模型产生了 {bad}/{n_draws} 个非有限结果（NaN/Inf）；"
            "通常意味着输入分布越过了模型定义域（如对负数开方/取对数）"
        )

    y = ctx.from_base(y_base)
    mean = float(np.mean(y))
    sd = float(np.std(y, ddof=1))
    median = float(np.median(y))
    q_lo, q_hi = (float(v) for v in np.quantile(y, [0.025, 0.975]))

    # ---- 逐输入相关性诊断（输出方差的相关份额 r^2） ----
    input_correlations: list[dict[str, float]] = []
    if sd > 0:
        yc = y - mean
        y_var = float(yc @ yc)
        for k, name in enumerate(names):
            xk = draws[k]
            if np.std(xk) > 0:
                xc = xk - xk.mean()
                r = float(xc @ yc / math.sqrt(float(xc @ xc) * y_var))
            else:
                r = 0.0
            input_correlations.append({"input": name, "pearson_r": r, "r_squared": r * r})
    else:
        input_correlations = [{"input": name, "pearson_r": 0.0, "r_squared": 0.0} for name in names]

    # ---- 收敛统计 ----
    kurt = float(((y - mean) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else 0.0
    se_mean = sd / math.sqrt(n_draws)
    se_sd = sd * math.sqrt(max(kurt + 2.0, 0.0) / (4.0 * n_draws)) if sd > 0 else 0.0
    half_width = max(q_hi - q_lo, _EPS)
    if sd > 0:
        f_lo = _gaussian_density(q_lo, y, sd)
        f_hi = _gaussian_density(q_hi, y, sd)
        f_lo = max(f_lo, 1e-12)
        f_hi = max(f_hi, 1e-12)
        se_q_lo = math.sqrt(0.025 * 0.975 / n_draws) / f_lo
        se_q_hi = math.sqrt(0.025 * 0.975 / n_draws) / f_hi
        se_q = max(se_q_lo, se_q_hi)
    else:
        se_q_lo = se_q_hi = se_q = 0.0

    denom_mean = max(abs(mean), sd)  # 均值接近 0 时以标准差为尺度，避免相对误差发散
    rel_mean = se_mean / denom_mean
    rel_sd = se_sd / sd if sd > 0 else 0.0
    rel_q = se_q / half_width

    reasons: list[str] = []
    if sd > 0:
        if rel_mean > convergence_tol:
            reasons.append(f"均值标准误相对值 {rel_mean:.3e} 超过容差 {convergence_tol:g}")
        if rel_sd > convergence_tol:
            reasons.append(f"标准差标准误相对值 {rel_sd:.3e} 超过容差 {convergence_tol:g}")
        if rel_q > convergence_tol:
            reasons.append(f"95% 分位标准误相对区间半宽 {rel_q:.3e} 超过容差 {convergence_tol:g}")
    converged = not reasons

    convergence = {
        "n": float(n_draws),
        "seed": float(0),  # 占位，由调用方覆盖（保持字段集中）
        "kurtosis_excess": kurt,
        "se_mean": se_mean,
        "se_std": se_sd,
        "se_quantile_low": se_q_lo,
        "se_quantile_high": se_q_hi,
        "rel_se_mean": rel_mean,
        "rel_se_std": rel_sd,
        "rel_se_quantile": rel_q,
        "tolerance": convergence_tol,
    }

    return McResult(
        mean=mean,
        std=sd,
        median=median,
        coverage_low=q_lo,
        coverage_high=q_hi,
        samples_min=float(np.min(y)),
        samples_max=float(np.max(y)),
        input_correlations=input_correlations,
        convergence=convergence,
        converged=converged,
        nonconvergence_reasons=reasons,
    )
