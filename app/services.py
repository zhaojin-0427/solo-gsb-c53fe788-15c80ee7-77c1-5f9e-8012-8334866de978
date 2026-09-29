"""业务编排：模型登记、快照校验、GUM+MC 计算管线。"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import uuid
from typing import Any

import numpy as np

from . import storage
from .engine import (
    build_correlation_matrix,
    run_gum,
    run_monte_carlo,
)
from .errors import ConflictError, ModelError, RequestError
from .expressions import compile_expression
from .models import EvaluationCreate, ModelCreate
from .units import build_unit_context, format_dimensionality, get_registry, parse_unit, to_declared_magnitude

#: 模型版本 -> (CompiledExpression, UnitContext)；登记时构建，进程内复用
_CONTEXT_CACHE: dict[str, tuple[Any, Any]] = {}

_VALUE_MATCH_REL = 1e-9
_VALUE_MATCH_ABS = 1e-12


# ---------------------------------------------------------------------------
# 规范化工具
# ---------------------------------------------------------------------------

def canonical_expression(source: str) -> str:
    """用 AST 重新输出表达式：等价的空格/括号写法规范化一致。"""
    try:
        return ast.unparse(ast.parse(source, mode="eval"))
    except SyntaxError as exc:
        raise ModelError(f"表达式语法错误: {exc.msg}") from exc


def _pairs(pairs: list[Any]) -> list[list[float]]:
    out = [[p.i, p.j, round(float(p.r), 15)] for p in pairs]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _input_names(inputs: list[Any]) -> tuple[str, ...]:
    names = [i.name for i in inputs]
    if len(set(names)) != len(names):
        raise ModelError("输入量名称必须唯一")
    return tuple(names)


def _validate_pair_names(pairs: list[Any], names: tuple[str, ...], where: str) -> None:
    name_set = set(names)
    seen: set[frozenset[str]] = set()
    for p in pairs:
        if p.i not in name_set or p.j not in name_set:
            raise RequestError(f"{where}中相关系数引用了不存在的输入量: {p.i!r}, {p.j!r}")
        if p.i == p.j:
            raise RequestError(f"{where}中输入量 {p.i!r} 不能与自身相关")
        key = frozenset((p.i, p.j))
        if key in seen:
            raise RequestError(f"{where}中相关系数 ({p.i!r}, {p.j!r}) 重复指定")
        seen.add(key)


# ---------------------------------------------------------------------------
# 模型登记
# ---------------------------------------------------------------------------

def register_model(req: ModelCreate) -> dict[str, Any]:
    names = _input_names(req.inputs)

    # 1. 表达式白名单校验
    compiled = compile_expression(req.expression, names)

    # 2. 单位解析 + 维度推导（拒绝量纲不一致的表达式与输出单位）
    declared_units = {}
    for spec in req.inputs:
        declared_units[spec.name] = parse_unit(spec.unit)
    _validate_pair_names(req.correlations, names, "模型")
    ctx = build_unit_context(compiled, declared_units, req.output_unit)

    # 3. 登记时刻即可校验相关矩阵半正定（使用登记的标准差）
    std_decl = np.array([i.standard_uncertainty for i in req.inputs], dtype=float)
    zero_std = {names[k] for k in np.where(std_decl == 0.0)[0]}
    corr_pairs = [(p.i, p.j, p.r) for p in req.correlations]
    build_correlation_matrix(names, corr_pairs, zero_std)

    reg = get_registry()
    expr_canonical = canonical_expression(req.expression)

    inputs_record = [
        {
            "name": s.name,
            "value": s.value,
            "unit": f"{declared_units[s.name]:~}",
            "standard_uncertainty": s.standard_uncertainty,
            "distribution": s.distribution,
        }
        for s in req.inputs
    ]
    correlations_record = _pairs(req.correlations)

    input_dimensions = {
        s.name: format_dimensionality(declared_units[s.name].dimensionality)
        for s in req.inputs
    }
    # 用输出单位推回输出维度（确保与表达式一致）
    output_dimensionality = format_dimensionality(parse_unit(req.output_unit).dimensionality)

    summary = {
        "model_version": req.version,
        "expression": expr_canonical,
        "expression_sha256": sha256_hex(expr_canonical),
        "variables": list(compiled.variables),
        "inputs": sorted(
            [
                {
                    "name": s.name,
                    "unit": f"{declared_units[s.name]:~}",
                    "dimensionality": input_dimensions[s.name],
                    "distribution": s.distribution,
                }
                for s in req.inputs
            ],
            key=lambda d: d["name"],
        ),
        "output_unit": req.output_unit,
        "output_dimensionality": output_dimensionality,
        "correlations": correlations_record,
        "whitelist": {
            "operators": ["+", "-", "*", "/", "**", "unary+", "unary-"],
            "functions": [
                "sqrt", "abs", "exp", "log", "log10",
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                "sinh", "cosh", "tanh",
            ],
            "constants": ["pi", "e"],
        },
    }

    row = {
        "version": req.version,
        "name": req.name,
        "description": req.description,
        "expression": req.expression,
        "expression_canonical": expr_canonical,
        "inputs": inputs_record,
        "output_unit": req.output_unit,
        "correlations": correlations_record,
        "summary": summary,
    }
    storage.insert_model(row)
    _CONTEXT_CACHE[req.version] = (compiled, ctx)
    return storage.get_model(req.version)


def _get_context(version: str) -> tuple[Any, Any]:
    cached = _CONTEXT_CACHE.get(version)
    if cached is not None:
        return cached
    record = storage.get_model(version)
    names = tuple(s["name"] for s in record["inputs"])
    compiled = compile_expression(record["expression"], names)
    declared_units = {s["name"]: parse_unit(s["unit"]) for s in record["inputs"]}
    ctx = build_unit_context(compiled, declared_units, record["output_unit"])
    _CONTEXT_CACHE[version] = (compiled, ctx)
    return compiled, ctx


# ---------------------------------------------------------------------------
# 计算请求
# ---------------------------------------------------------------------------

def _match_value(a: float, b: float) -> bool:
    return abs(a - b) <= max(_VALUE_MATCH_ABS, _VALUE_MATCH_REL * max(abs(a), abs(b)))


def _reconcile_snapshot(req: EvaluationCreate, record: dict[str, Any]) -> dict[str, Any]:
    """把请求输入快照换算到模型声明单位，并与登记的不可变模型逐项核对。

    返回规范化快照（声明单位下的数值）。单位允许等价写法（如 ``V`` 与 ``mV``），
    其余字段（值、标准不确定度、分布、相关系数）必须与登记一致。
    """
    req_by_name = {s.name: s for s in req.inputs}
    reg_names = [s["name"] for s in record["inputs"]]
    if set(req_by_name) != set(reg_names):
        missing = sorted(set(reg_names) - set(req_by_name))
        extra = sorted(set(req_by_name) - set(reg_names))
        raise RequestError(f"输入快照与模型不一致（缺失 {missing}，多余 {extra}）")

    declared_units = {s["name"]: parse_unit(s["unit"]) for s in record["inputs"]}
    normalized: list[dict[str, Any]] = []
    for reg in record["inputs"]:
        snap = req_by_name[reg["name"]]
        unit = declared_units[reg["name"]]
        try:
            value_decl = to_declared_magnitude(snap.value, snap.unit, unit)
            std_decl = to_declared_magnitude(snap.standard_uncertainty, snap.unit, unit)
        except RequestError:
            raise

        if not _match_value(value_decl, reg["value"]):
            raise RequestError(
                f"输入 {reg['name']!r} 的名义值与登记模型不符: "
                f"快照换算后 {value_decl!r}，登记 {reg['value']!r}"
            )
        if not _match_value(std_decl, reg["standard_uncertainty"]):
            raise RequestError(
                f"输入 {reg['name']!r} 的标准不确定度与登记模型不符: "
                f"快照换算后 {std_decl!r}，登记 {reg['standard_uncertainty']!r}"
            )
        if snap.distribution != reg["distribution"]:
            raise RequestError(
                f"输入 {reg['name']!r} 的分布与登记模型不符: "
                f"{snap.distribution} != {reg['distribution']}"
            )
        normalized.append({
            "name": reg["name"],
            "value": reg["value"],
            "unit": f"{unit:~}",
            "standard_uncertainty": reg["standard_uncertainty"],
            "distribution": reg["distribution"],
        })

    normalized.sort(key=lambda d: d["name"])

    # 相关系数核对（规范化为有序对集合）
    req_pairs = {frozenset((p.i, p.j)): round(float(p.r), 12) for p in req.correlations}
    reg_pairs = {frozenset((t[0], t[1])): float(t[2]) for t in record["correlations"]}
    if set(req_pairs) != set(reg_pairs):
        raise RequestError("相关系数对的集合与登记模型不一致")
    for key, r in req_pairs.items():
        if not _match_value(r, reg_pairs[key]):
            raise RequestError(f"相关系数 {sorted(key)} 与登记模型不符: {r} != {reg_pairs[key]}")

    return {
        "inputs": normalized,
        "correlations": copy.deepcopy(record["correlations"]),
    }


def submit_evaluation(req: EvaluationCreate) -> dict[str, Any]:
    record = storage.get_model(req.model_version)  # 不存在 -> 404
    compiled, ctx = _get_context(req.model_version)

    # 1. 快照与不可变模型核对（含单位换算、维度兼容）
    snapshot = _reconcile_snapshot(req, record)

    # 2. 请求指纹：模型版本 + 输入快照 + N + 种子 + 容差 + 幂等键
    fingerprint = {
        "model_version": req.model_version,
        "snapshot": snapshot,
        "n": req.n,
        "seed": req.seed,
        "convergence_tolerance": req.convergence_tolerance,
        "idempotency_key": req.idempotency_key,
    }
    request_hash = sha256_hex(canonical_json(fingerprint))

    # 3. 幂等快路径
    prior = storage.find_evaluation_by_idempotency_key(req.idempotency_key)
    if prior is not None:
        if prior["request_hash"] != request_hash:
            raise ConflictError(
                f"幂等键 {req.idempotency_key!r} 已绑定到不同的计算请求（evaluation {prior['id']}）"
            )
        prior["cached"] = True
        return prior

    names = tuple(s["name"] for s in record["inputs"])
    distributions = tuple(
        next(s["distribution"] for s in record["inputs"] if s["name"] == name)
        for name in names
    )

    # 4. 基本单位空间
    x_decl = {s["name"]: s["value"] for s in record["inputs"]}
    std_decl = {s["name"]: s["standard_uncertainty"] for s in record["inputs"]}
    x_base = ctx.nominal_to_base(x_decl)
    std_base_arr = np.array([ctx.std_to_base(std_decl)[name] for name in names], dtype=float)
    zero_std = {name for name in names if std_decl[name] == 0.0}

    corr = build_correlation_matrix(
        names, [(t[0], t[1], float(t[2])) for t in record["correlations"]], zero_std
    )

    # 5. GUM 一阶
    gum = run_gum(compiled, names, x_base, std_base_arr, corr, ctx)

    # 6. 蒙特卡洛（固定种子）；不收敛 -> 抛错，不保存任何结果
    mc = run_monte_carlo(
        compiled=compiled,
        names=names,
        distributions=distributions,
        x_base=x_base,
        std_base=std_base_arr,
        corr=corr,
        n_draws=req.n,
        seed=req.seed,
        ctx=ctx,
        convergence_tol=req.convergence_tolerance,
    )
    if not mc.converged:
        from .errors import ConvergenceError

        raise ConvergenceError(
            f"蒙特卡洛 N={req.n} 未通过收敛判据（可增大 N 后以新幂等键重试）: "
            + "; ".join(mc.nonconvergence_reasons),
        )
    mc.convergence["seed"] = float(req.seed)

    # 7. 组装结果（仅在计算全部成功后写库，且写库是唯一的“保存”动作）
    result = {
        "model_version": req.model_version,
        "model_summary": record["summary"],
        "method": {
            "gum": "GUM 一阶线性化（Guide to the Expression of Uncertainty in Measurement）",
            "monte_carlo": "GUM Supplement 1 自适应风格蒙特卡洛（Gaussian copula + 边际分布）",
            "coverage": 0.95,
        },
        "snapshot": snapshot,
        "parameters": {
            "n": req.n,
            "seed": req.seed,
            "convergence_tolerance": req.convergence_tolerance,
        },
        "combined": {
            "gum": {
                "value": _f(gum.y),
                "standard_uncertainty": _f(gum.uc),
                "coverage_interval": {"low": _f(gum.coverage_low), "high": _f(gum.coverage_high)},
                "coverage_factor": _f(gum.k),
                "unit": req_output_unit(record),
            },
            "monte_carlo": {
                "mean": _f(mc.mean),
                "standard_uncertainty": _f(mc.std),
                "median": _f(mc.median),
                "coverage_interval": {"low": _f(mc.coverage_low), "high": _f(mc.coverage_high)},
                "samples_range": {"min": _f(mc.samples_min), "max": _f(mc.samples_max)},
                "unit": req_output_unit(record),
            },
        },
        "sensitivities": {
            name: {"coefficient": _f(c), "unit": f"{record['output_unit']}/{s_unit(name, record)}"}
            for name, c in gum.sensitivities.items()
        },
        "contributions": {
            "gum": [
                {
                    "source": c["input"],
                    "kind": c["kind"],
                    "variance_contribution": _f(float(c["variance_contribution"])),
                    "fraction": _f(float(c["fraction"])),
                }
                for c in gum.contributions
            ],
            "monte_carlo": [
                {
                    "source": d["input"],
                    "pearson_r": _f(d["pearson_r"]),
                    "r_squared": _f(d["r_squared"]),
                }
                for d in mc.input_correlations
            ],
        },
        "convergence": {**{k: _f(v) for k, v in mc.convergence.items()}, "converged": True},
    }

    eval_id = uuid.uuid4().hex
    stored = storage.insert_evaluation(
        eval_id=eval_id,
        request_hash=request_hash,
        idempotency_key=req.idempotency_key,
        model_version=req.model_version,
        inputs_json=canonical_json(snapshot),
        n=req.n,
        seed=req.seed,
        tol=req.convergence_tolerance,
        result_json=canonical_json(result),
    )
    out = stored["evaluation"]
    out["cached"] = stored["cached"]
    return out


def req_output_unit(record: dict[str, Any]) -> str:
    return record["output_unit"]


def s_unit(name: str, record: dict[str, Any]) -> str:
    return next(s["unit"] for s in record["inputs"] if s["name"] == name)


def _f(x: Any) -> float:
    """统一转 float（剔除 numpy 标量，保证 JSON 可序列化）。"""
    return round(float(x), 15)
