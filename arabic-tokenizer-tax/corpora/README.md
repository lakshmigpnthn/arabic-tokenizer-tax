# The Arabic Tokenizer Tax

**Measure how much more your Arabic-heavy AI workload costs on frontier cloud LLMs compared to equivalent English content — and project the gap as a monthly bill on a realistic enterprise workload.**

A runnable companion to the article *The Tokenizer Tax: The Hidden Cost of Arabic AI Workloads*.

---

## Try it now (no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-HANDLE/arabic-tokenizer-tax/blob/main/notebooks/arabic_tokenizer_tax.ipynb)

Click the badge above, then **Runtime → Run all**. First run takes about 60–90 seconds to install dependencies. After that you will see:

1. **The token-count gap** across four tokenizers (`cl100k_base`, `o200k_base`, Qwen 2.5, Llama 3) on a realistic AML review document in English vs Arabic.
2. **The projected monthly cost** for a UAE bank running 500,000 alerts a month, in both languages, across GPT-4o, Claude, and Gemini.
3. **A cell where you can paste your own text** and run the same comparison on your corpus.

No API keys, no Hugging Face logins, no local setup required for the default tokenizers.

---

## What is in this repo

```
arabic-tokenizer-tax/
├── notebooks/
│   └── arabic_tokenizer_tax.ipynb     # The Colab notebook (start here)
├── src/                                # Python library + CLI + Streamlit app
│   ├── tokenizers.py                   # Tokenizer wrappers
│   ├── compare.py                      # CLI comparison tool
│   ├── cost.py                         # Cost projection
│   └── app.py                          # Streamlit web app
├── corpora/                            # Sample paired EN/AR text
├── examples/                           # Workload and pricing YAML
└── requirements.txt
```

The notebook is self-contained and is what most readers will run. The `src/` directory is for people who want to build on it.

---

## What this measures

For each tokenizer, the notebook counts how many tokens the same business content produces in English vs Arabic. The ratio `AR tokens / EN tokens` is what you actually pay extra for on per-token cloud LLM pricing.

The cost projection takes that ratio and applies it to a workload you can edit (volume per month, baseline tokens in/out per call) and pricing you can edit (per-model $/1M tokens for input and output).

---

## Honest caveats

- **Pricing is illustrative.** Vendor pricing changes, varies by region and tier, and may differ from your contracted rates. Update the pricing dict in the notebook with current figures before quoting anything to a CFO.
- **Tokenizer behaviour varies by content.** A legal contract, a customer chat transcript, and a regulatory filing will tokenize differently. Test on a representative sample of your own corpus.
- **Claude and Gemini do not ship public tokenizer libraries.** The notebook does not include them directly. For Claude, use Anthropic's `count_tokens` API. For Gemini, use Google's `count_tokens` method.
- **Jais and AceGPT require Hugging Face access requests.** They are gated. The notebook includes optional cells for adding them after you have logged in.
- **This is a measurement tool, not a model recommendation.** It tells you what your token cost gap is. The decision about which model to use depends on accuracy, latency, sovereignty, and cost — only one of which this measures.

---

## Why this exists

I wrote a short LinkedIn article on the Arabic tokenizer tax in Gulf enterprise AI workloads. The most useful response I got was effectively *"show me the numbers on my data."* This repo is the tool for doing that.

If you ran the notebook on your own corpus and got results that surprised you — or contradict the article — open an issue. If the methodology has a flaw, even better, tell me about it.

---

## License

MIT.
