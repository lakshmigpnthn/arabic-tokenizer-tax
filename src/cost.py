"""
Cost projection for Arabic vs English LLM workloads.

Takes:
- a measured AR/EN token ratio (from `compare`)
- a workload definition (volume, baseline tokens in/out)
- a pricing definition (per-model $/1M tokens)

Outputs the projected monthly cost in both languages and the delta.

Usage:
    python -m src.cost \\
        --en corpora/sample_en.txt \\
        --ar corpora/sample_ar.txt \\
        --workload examples/uae_bank_aml.yaml \\
        --pricing examples/pricing.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .tokenizers import count_all


def _read(p: str | None) -> str:
    if not p:
        sys.exit("[error] missing required argument")
    path = Path(p)
    if not path.exists():
        sys.exit(f"[error] file not found: {p}")
    return path.read_text(encoding="utf-8")


def project_cost(
    en_text: str,
    ar_text: str,
    workload: dict,
    pricing: dict,
    tokenizer: str = "cl100k_base",
) -> dict:
    """
    Project monthly cost for a workload in English vs Arabic.

    workload schema:
        volume_per_month: int            # e.g. 500_000
        input_tokens_en_baseline: int    # baseline input tokens per call (English)
        output_tokens_en_baseline: int   # baseline output tokens per call (English)
        sample_text_basis: str           # "input" or "output" or "both"

    pricing schema:
        models:
          gpt-4o:
            input_per_1m: 2.50
            output_per_1m: 10.00
          claude-sonnet-4:
            input_per_1m: 3.00
            output_per_1m: 15.00
    """
    en_count = count_all(en_text, [tokenizer]).get(tokenizer, 0)
    ar_count = count_all(ar_text, [tokenizer]).get(tokenizer, 0)
    if en_count <= 0 or ar_count <= 0:
        sys.exit(f"[error] tokenization failed for {tokenizer}")

    ratio = ar_count / en_count

    en_in = workload["input_tokens_en_baseline"]
    en_out = workload["output_tokens_en_baseline"]
    volume = workload["volume_per_month"]

    ar_in = en_in * ratio
    ar_out = en_out * ratio

    results = {
        "tokenizer": tokenizer,
        "ar_en_ratio": ratio,
        "per_call": {
            "en_input": en_in,
            "en_output": en_out,
            "ar_input": ar_in,
            "ar_output": ar_out,
        },
        "monthly_tokens": {
            "en_input": en_in * volume,
            "en_output": en_out * volume,
            "ar_input": ar_in * volume,
            "ar_output": ar_out * volume,
        },
        "models": {},
    }

    for model_name, p in pricing["models"].items():
        en_cost = (
            (en_in * volume / 1_000_000) * p["input_per_1m"]
            + (en_out * volume / 1_000_000) * p["output_per_1m"]
        )
        ar_cost = (
            (ar_in * volume / 1_000_000) * p["input_per_1m"]
            + (ar_out * volume / 1_000_000) * p["output_per_1m"]
        )
        results["models"][model_name] = {
            "en_monthly_usd": en_cost,
            "ar_monthly_usd": ar_cost,
            "delta_monthly_usd": ar_cost - en_cost,
            "delta_pct": ((ar_cost - en_cost) / en_cost * 100) if en_cost else 0,
        }

    return results


def _print_report(r: dict, workload: dict) -> None:
    print()
    print(f"Workload: {workload.get('name', 'unnamed')}")
    print(f"  Volume: {workload['volume_per_month']:,} calls/month")
    print(f"  EN baseline: {workload['input_tokens_en_baseline']:,} in / "
          f"{workload['output_tokens_en_baseline']:,} out per call")
    print(f"  Tokenizer used for ratio: {r['tokenizer']}")
    print(f"  Measured AR/EN ratio: {r['ar_en_ratio']:.2f}x")
    print()
    print(f"{'Model':<24}{'EN $/mo':>14}{'AR $/mo':>14}{'Delta':>14}{'Delta %':>10}")
    print("-" * 76)
    for model, m in r["models"].items():
        print(
            f"{model:<24}"
            f"{m['en_monthly_usd']:>14,.0f}"
            f"{m['ar_monthly_usd']:>14,.0f}"
            f"{m['delta_monthly_usd']:>14,.0f}"
            f"{m['delta_pct']:>9.0f}%"
        )
    print()
    print("Pricing is configurable — see examples/pricing.yaml. Replace with your "
          "contracted rates before quoting any specific figure.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Project Arabic vs English LLM workload cost.")
    ap.add_argument("--en", required=True, help="Path to English text file")
    ap.add_argument("--ar", required=True, help="Path to Arabic text file")
    ap.add_argument("--workload", required=True, help="Path to workload YAML")
    ap.add_argument("--pricing", required=True, help="Path to pricing YAML")
    ap.add_argument("--tokenizer", default="cl100k_base", help="Tokenizer to use for the ratio")
    args = ap.parse_args()

    en = _read(args.en)
    ar = _read(args.ar)
    workload = yaml.safe_load(_read(args.workload))
    pricing = yaml.safe_load(_read(args.pricing))

    report = project_cost(en, ar, workload, pricing, tokenizer=args.tokenizer)
    _print_report(report, workload)


if __name__ == "__main__":
    main()
