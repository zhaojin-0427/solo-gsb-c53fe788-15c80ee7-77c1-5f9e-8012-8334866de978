#!/usr/bin/env bash
# 最小端到端示例：登记 P=U*I 模型 -> 提交计算 -> 幂等重放 -> 查询
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

echo "== 健康检查 =="
curl -fsS "$BASE/health" && echo

echo "== 1. 登记模型 power-dc-v1（已存在则忽略 409）=="
curl -sS -X POST "$BASE/api/v1/models" -H 'Content-Type: application/json' -d '{
  "version": "power-dc-v1",
  "name": "直流功率 P = U*I",
  "expression": "U * I",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "output_unit": "W",
  "correlations": [{"i": "U", "j": "I", "r": 0.5}]
}' -o /tmp/model_resp.json -w "HTTP %{http_code}\n" || true

echo "== 2. 提交计算（N=20000，固定种子）=="
curl -sS -X POST "$BASE/api/v1/evaluations" -H 'Content-Type: application/json' -d '{
  "model_version": "power-dc-v1",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "correlations": [{"i": "U", "j": "I", "r": 0.5}],
  "n": 20000,
  "seed": 20260929,
  "idempotency_key": "demo-power-0001"
}' -o /tmp/eval_resp.json -w "HTTP %{http_code}\n"

python3 - <<'PY'
import json
d = json.load(open('/tmp/eval_resp.json'))
g, m = d["result"]["combined"]["gum"], d["result"]["combined"]["monte_carlo"]
print(f"GUM : {g['value']:.4f} ± {g['standard_uncertainty']:.4f} {g['unit']} "
      f"95%=[{g['coverage_interval']['low']:.4f}, {g['coverage_interval']['high']:.4f}]")
print(f"MC  : {m['mean']:.4f} ± {m['standard_uncertainty']:.4f} {m['unit']} "
      f"95%=[{m['coverage_interval']['low']:.4f}, {m['coverage_interval']['high']:.4f}]")
print("evaluation id:", d["id"], "| cached:", d.get("cached"))
PY

echo "== 3. 幂等重放（期望 HTTP 200, cached=true）=="
curl -sS -X POST "$BASE/api/v1/evaluations" -H 'Content-Type: application/json' -d '{
  "model_version": "power-dc-v1",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "correlations": [{"i": "U", "j": "I", "r": 0.5}],
  "n": 20000,
  "seed": 20260929,
  "idempotency_key": "demo-power-0001"
}' -o /dev/null -w "HTTP %{http_code}\n"

echo "== 4. 查询 =="
curl -fsS "$BASE/api/v1/idempotency/demo-power-0001" -o /dev/null -w "by key: HTTP %{http_code}\n"
curl -fsS "$BASE/api/v1/models/power-dc-v1" -o /dev/null -w "model  : HTTP %{http_code}\n"
