"""
Tokenizer wrappers for the Arabic Tokenizer Tax demo.

Each tokenizer is exposed through a common interface:
    count(text: str) -> int

Tokenizers are loaded lazily — only when first used — so users don't pay
the startup cost for tokenizers they aren't comparing against.

Notes on accuracy:
- OpenAI tokenizers (cl100k_base, o200k_base) via `tiktoken` are exact.
- Open-weight tokenizers (Qwen, Jais, AceGPT) via `transformers` are exact for
  those models.
- Claude and Gemini tokenizers are approximations. Anthropic and Google do not
  publish standalone tokenizer libraries the way OpenAI does. For Claude, we
  use Anthropic's published API token-counting endpoint as the ground truth
  when an API key is provided, and fall back to a cl100k-based estimate
  otherwise. For Gemini, we use Google's `count_tokens` API where available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import os


@dataclass
class TokenizerInfo:
    name: str
    provider: str
    kind: str  # "frontier" | "open-weight" | "arabic-tuned"
    notes: str
    count: Callable[[str], int]


def _lazy(loader: Callable[[], Callable[[str], int]]) -> Callable[[str], int]:
    """Wrap a loader so the underlying tokenizer is only loaded on first call."""
    cache: Dict[str, Callable[[str], int]] = {}

    def wrapped(text: str) -> int:
        if "fn" not in cache:
            cache["fn"] = loader()
        return cache["fn"](text)

    return wrapped


# --- OpenAI (tiktoken) -------------------------------------------------------

def _load_tiktoken(name: str) -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding(name)
    return lambda text: len(enc.encode(text))


# --- Hugging Face transformers ----------------------------------------------

def _load_hf(model_id: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


# --- Claude (Anthropic) ------------------------------------------------------

def _load_claude() -> Callable[[str], int]:
    """
    Use Anthropic's count_tokens API if a key is available; otherwise fall back
    to a cl100k_base approximation with a note that it is approximate.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)

            def count(text: str) -> int:
                resp = client.messages.count_tokens(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": text}],
                )
                return resp.input_tokens

            return count
        except Exception:
            pass

    # Fallback: cl100k approximation
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


# --- Gemini (Google) ---------------------------------------------------------

def _load_gemini() -> Callable[[str], int]:
    """
    Use Google's count_tokens if `google-generativeai` is installed and a key
    is set; otherwise fall back to a cl100k approximation.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            def count(text: str) -> int:
                return model.count_tokens(text).total_tokens

            return count
        except Exception:
            pass

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


# --- Registry ----------------------------------------------------------------

TOKENIZERS: Dict[str, TokenizerInfo] = {
    "cl100k_base": TokenizerInfo(
        name="cl100k_base",
        provider="OpenAI",
        kind="frontier",
        notes="Used by GPT-4 and GPT-4o family. Exact (tiktoken).",
        count=_lazy(lambda: _load_tiktoken("cl100k_base")),
    ),
    "o200k_base": TokenizerInfo(
        name="o200k_base",
        provider="OpenAI",
        kind="frontier",
        notes="Used by newer GPT-4o variants. Exact (tiktoken).",
        count=_lazy(lambda: _load_tiktoken("o200k_base")),
    ),
    "claude": TokenizerInfo(
        name="claude",
        provider="Anthropic",
        kind="frontier",
        notes="Uses Anthropic count_tokens API if ANTHROPIC_API_KEY is set, else cl100k approximation.",
        count=_lazy(_load_claude),
    ),
    "gemini": TokenizerInfo(
        name="gemini",
        provider="Google",
        kind="frontier",
        notes="Uses Google count_tokens API if GOOGLE_API_KEY is set, else cl100k approximation.",
        count=_lazy(_load_gemini),
    ),
    "qwen2.5": TokenizerInfo(
        name="qwen2.5",
        provider="Alibaba",
        kind="open-weight",
        notes="Multilingual; some Arabic-tuned variants available.",
        count=_lazy(lambda: _load_hf("Qwen/Qwen2.5-7B")),
    ),
    "jais-13b": TokenizerInfo(
        name="jais-13b",
        provider="Inception / G42 / MBZUAI",
        kind="arabic-tuned",
        notes="Arabic-native model. Tokenizer download is gated; request access on Hugging Face first.",
        count=_lazy(lambda: _load_hf("inceptionai/jais-13b")),
    ),
    "acegpt": TokenizerInfo(
        name="acegpt",
        provider="KAUST",
        kind="arabic-tuned",
        notes="Arabic-tuned Llama. Tokenizer download is gated on Hugging Face.",
        count=_lazy(lambda: _load_hf("FreedomIntelligence/AceGPT-7B")),
    ),
}


def available_tokenizers() -> Dict[str, TokenizerInfo]:
    return TOKENIZERS


def count_all(text: str, names: Optional[list[str]] = None) -> Dict[str, int]:
    """Return a dict of {tokenizer_name: token_count} for the given text."""
    selected = names or list(TOKENIZERS.keys())
    results: Dict[str, int] = {}
    for name in selected:
        info = TOKENIZERS.get(name)
        if not info:
            continue
        try:
            results[name] = info.count(text)
        except Exception as e:
            results[name] = -1  # signal failure
            print(f"[warn] {name} failed: {e}")
    return results
