"""Offline evaluation against the real LLM (memory backend, fixed demo data).

    CA_LLM_API_KEY=... python eval/run_eval.py [--min-pass 0.9] [--repeat 3]

Gate prompt and model changes on this: it catches regressions in routing, grounding,
escalation and member isolation before they ship. Grow the case list from reviewed case_runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careagents.config import Settings  # noqa: E402
from careagents.container import build_container  # noqa: E402


def check(case: dict, r) -> list[str]:  # type: ignore[no-untyped-def]
    problems = []
    if "expect_outcome" in case and r.outcome != case["expect_outcome"]:
        problems.append(f"outcome {r.outcome} != {case['expect_outcome']}")
    if "expect_outcome_in" in case and r.outcome not in case["expect_outcome_in"]:
        problems.append(f"outcome {r.outcome} not in {case['expect_outcome_in']}")
    missing = set(case.get("expect_agents", [])) - set(r.routed_to)
    if missing:
        problems.append(f"agents not routed: {sorted(missing)}")
    extra = set(case.get("forbid_agents", [])) & set(r.routed_to)
    if extra:
        problems.append(f"forbidden agents routed: {sorted(extra)}")
    text = r.answer.lower()
    for pattern in case.get("must_contain", []):
        if not any(alt.lower() in text for alt in pattern.split("|")):
            problems.append(f"answer missing '{pattern}'")
    for s in case.get("must_not_contain", []):
        if s.lower() in text or any(s.lower() in f.value.lower() for f in r.facts):
            problems.append(f"leaked or wrong: '{s}'")
    keys = {f.key for f in r.facts}
    for k in case.get("required_facts", []):
        if k not in keys:
            problems.append(f"fact missing: {k}")
    if re.search(r"\$\s?\d", r.answer) and r.outcome == "answered" and not r.facts and not r.sources:
        problems.append("amounts without any grounding")
    return problems


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pass", type=float, default=0.9)
    ap.add_argument("--repeat", type=int, default=1, help="run each case N times to measure consistency")
    ap.add_argument("--only")
    args = ap.parse_args()

    settings = Settings(env="test", data_backend="memory", documents_dir=ROOT / "data" / "documents")
    if not settings.llm_api_key.get_secret_value():
        print("CA_LLM_API_KEY is required for evals", file=sys.stderr)
        return 2
    container = await build_container(settings, with_queue=False)
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    results, latencies, tokens = [], [], []
    for case in cases:
        for i in range(args.repeat):
            r = await container.runtime.handle(case["question"], case["member"], "eval", f"eval-{case['id']}-{i}")
            problems = check(case, r)
            results.append(not problems)
            latencies.append(r.latency_ms)
            tokens.append(r.usage.prompt_tokens + r.usage.completion_tokens)
            status = "PASS" if not problems else "FAIL"
            print(f"{status} {case['id']}#{i} [{r.outcome}, {r.latency_ms}ms, {r.usage.llm_calls} calls] "
                  + ("; ".join(problems) if problems else ""))
    await container.aclose()

    rate = sum(results) / len(results)
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    print(f"\npass rate {rate:.0%} ({sum(results)}/{len(results)}) | latency p50 "
          f"{statistics.median(latencies):.0f}ms p95 {p95}ms | avg tokens {statistics.mean(tokens):.0f}")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
