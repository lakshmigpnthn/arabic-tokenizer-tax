"""
CLI: compare English and Arabic text across tokenizers.

Usage:
    python -m src.compare --en corpora/sample_en.txt --ar corpora/sample_ar.txt
    python -m src.compare --en-text "compliance review" --ar-text "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .tokenizers import available_tokenizers, count_all


def _read(arg_text: str | None, arg_file: str | None, label: str) -> str:
    if arg_text:
        return arg_text
    if arg_file:
        p = Path(arg_file)
        if not p.exists():
            sys.exit(f"[error] {label} file not found: {arg_file}")
        return p.read_text(encoding="utf-8")
    sys.exit(f"[error] provide --{label}-text or --{label} <file>")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare English vs Arabic token counts across tokenizers.")
    ap.add_argument("--en", help="Path to English text file")
    ap.add_argument("--ar", help="Path to Arabic text file")
    ap.add_argument("--en-text", help="English text (inline)")
    ap.add_argument("--ar-text", help="Arabic text (inline)")
    ap.add_argument(
        "--tokenizers",
        nargs="*",
        default=None,
        help="Subset of tokenizer names to use (default: all)",
    )
    args = ap.parse_args()

    en = _read(args.en_text, args.en, "en")
    ar = _read(args.ar_text, args.ar, "ar")

    names = args.tokenizers or list(available_tokenizers().keys())

    print(f"\nEnglish chars: {len(en):,}   Arabic chars: {len(ar):,}\n")
    print(f"{'Tokenizer':<18}{'EN tokens':>12}{'AR tokens':>12}{'AR/EN':>10}")
    print("-" * 52)

    en_counts = count_all(en, names)
    ar_counts = count_all(ar, names)

    for name in names:
        e = en_counts.get(name, -1)
        a = ar_counts.get(name, -1)
        if e <= 0 or a <= 0:
            print(f"{name:<18}{'n/a':>12}{'n/a':>12}{'n/a':>10}")
            continue
        ratio = a / e
        print(f"{name:<18}{e:>12,}{a:>12,}{ratio:>9.2f}x")

    print()


if __name__ == "__main__":
    main()
