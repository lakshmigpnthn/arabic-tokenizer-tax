"""
Streamlit UI for the Arabic Tokenizer Tax demo.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from src.tokenizers import available_tokenizers, count_all
from src.cost import project_cost


st.set_page_config(
    page_title="The Arabic Tokenizer Tax",
    page_icon="🪙",
    layout="wide",
)

st.title("The Arabic Tokenizer Tax")
st.markdown(
    "Measure how much more your Arabic-heavy AI workload costs on frontier cloud LLMs "
    "compared to equivalent English content. Paste paired text below, run the comparison, "
    "and project a monthly bill against your workload."
)

# --- Tokenizer selector ------------------------------------------------------

tokenizers_info = available_tokenizers()
with st.sidebar:
    st.header("Tokenizers")
    selected = st.multiselect(
        "Choose tokenizers to compare",
        options=list(tokenizers_info.keys()),
        default=["cl100k_base", "o200k_base", "qwen2.5"],
    )
    st.caption(
        "Arabic-tuned tokenizers (Jais, AceGPT) are gated on Hugging Face. "
        "Run `huggingface-cli login` after requesting access."
    )

    st.header("Notes")
    for name in selected:
        info = tokenizers_info[name]
        st.caption(f"**{info.name}** ({info.kind}): {info.notes}")

# --- Input ------------------------------------------------------------------

col_en, col_ar = st.columns(2)
with col_en:
    st.subheader("English text")
    en_text = st.text_area(
        "Paste English content",
        height=240,
        placeholder="Paste English content here. For meaningful results, use a representative sample of your real workload.",
    )

with col_ar:
    st.subheader("Arabic text")
    ar_text = st.text_area(
        "Paste Arabic content of equivalent meaning",
        height=240,
        placeholder="Paste the Arabic equivalent here. Equivalent meaning matters more than character count.",
    )

run = st.button("Compare", type="primary", disabled=not (en_text and ar_text and selected))

# --- Comparison -------------------------------------------------------------

if run:
    with st.spinner("Tokenizing — first run may download tokenizer files..."):
        en_counts = count_all(en_text, selected)
        ar_counts = count_all(ar_text, selected)

    rows = []
    for name in selected:
        e = en_counts.get(name, -1)
        a = ar_counts.get(name, -1)
        if e <= 0 or a <= 0:
            rows.append({
                "Tokenizer": name,
                "EN tokens": "n/a",
                "AR tokens": "n/a",
                "AR/EN ratio": "n/a",
            })
        else:
            rows.append({
                "Tokenizer": name,
                "EN tokens": e,
                "AR tokens": a,
                "AR/EN ratio": round(a / e, 2),
            })
    df = pd.DataFrame(rows)
    st.subheader("Token counts")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Persist for cost section
    st.session_state["en_text"] = en_text
    st.session_state["ar_text"] = ar_text
    st.session_state["last_counts"] = (en_counts, ar_counts)

# --- Cost projection --------------------------------------------------------

st.divider()
st.header("Cost projection")
st.markdown(
    "Define your workload and pricing below. Defaults model a UAE bank running AML alerts; "
    "edit them to match your case. Pricing is in USD per 1M tokens."
)

c1, c2, c3 = st.columns(3)
with c1:
    volume = st.number_input("Monthly call volume", min_value=1, value=500_000, step=10_000)
with c2:
    en_in = st.number_input("EN input tokens per call (baseline)", min_value=1, value=2_000, step=100)
with c3:
    en_out = st.number_input("EN output tokens per call (baseline)", min_value=1, value=500, step=50)

st.subheader("Pricing (USD per 1M tokens)")
st.caption(
    "These are illustrative defaults. Update with your contracted rates or current published "
    "rates before quoting any specific figure."
)
pricing_default = {
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "claude-sonnet-4": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "gemini-1.5-pro": {"input_per_1m": 1.25, "output_per_1m": 5.00},
}

pricing_df = pd.DataFrame(
    [
        {"Model": m, "Input $/1M": v["input_per_1m"], "Output $/1M": v["output_per_1m"]}
        for m, v in pricing_default.items()
    ]
)
edited = st.data_editor(pricing_df, hide_index=True, num_rows="dynamic", use_container_width=True)

tokenizer_for_ratio = st.selectbox(
    "Tokenizer to use for the AR/EN ratio",
    options=selected if selected else ["cl100k_base"],
)

if st.button("Project cost", disabled=not (st.session_state.get("en_text") and st.session_state.get("ar_text"))):
    workload = {
        "name": "custom",
        "volume_per_month": int(volume),
        "input_tokens_en_baseline": int(en_in),
        "output_tokens_en_baseline": int(en_out),
    }
    pricing = {
        "models": {
            row["Model"]: {
                "input_per_1m": float(row["Input $/1M"]),
                "output_per_1m": float(row["Output $/1M"]),
            }
            for _, row in edited.iterrows()
            if row["Model"]
        }
    }

    with st.spinner("Projecting..."):
        report = project_cost(
            st.session_state["en_text"],
            st.session_state["ar_text"],
            workload,
            pricing,
            tokenizer=tokenizer_for_ratio,
        )

    st.metric("Measured AR/EN ratio", f"{report['ar_en_ratio']:.2f}x")

    cost_rows = []
    for model, m in report["models"].items():
        cost_rows.append({
            "Model": model,
            "EN monthly ($)": round(m["en_monthly_usd"], 0),
            "AR monthly ($)": round(m["ar_monthly_usd"], 0),
            "Delta ($)": round(m["delta_monthly_usd"], 0),
            "Delta (%)": round(m["delta_pct"], 0),
        })
    st.dataframe(pd.DataFrame(cost_rows), hide_index=True, use_container_width=True)

    st.info(
        "These are projections, not quotes. Pricing changes, tokenizer behaviour varies by content, "
        "and your actual usage patterns may differ from the baseline. Use these numbers to start a "
        "conversation with your vendor or your CFO, not to end one."
    )
