import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render_sidebar_header():
    st.markdown(
        """
        <style>
        /* Compact, proportional layout & typography across all experiment pages */
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
            line-height: 1.25 !important;
            margin-bottom: 0.35rem !important;
        }
        h2 {
            font-size: 1.15rem !important;
            line-height: 1.25 !important;
        }
        h3 {
            font-size: 1.0rem !important;
            line-height: 1.25 !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.25rem !important;
        }
        /* Fix st.metric value truncation (...) across all 4-column metric bars */
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            font-size: 1.12rem !important;
            line-height: 1.25 !important;
            white-space: normal !important;
            overflow-wrap: break-word !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] p {
            font-size: 0.80rem !important;
            line-height: 1.2 !important;
        }
        /* Tighter slider & widget labels */
        .stSlider label p, .stNumberInput label p, .stCheckbox label p {
            font-size: 0.85rem !important;
        }
        /* Compact body markdown & alert boxes */
        .stMarkdown p, .stAlert p {
            font-size: 0.88rem !important;
            line-height: 1.4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def style_cro_axes(ax, title="", xlabel="Time (ms)", ylabel="Voltage (V)"):
    """Apply clean Dual-Trace CRO / laboratory oscilloscope styling."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="#666666", linewidth=0.9, linestyle="-")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.minorticks_on()

def format_eng(value, unit=""):
    """Format engineering values cleanly with SI prefixes."""
    if value == 0:
        return f"0 {unit}".strip()
    prefixes = [
        (1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""),
        (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")
    ]
    abs_val = abs(value)
    for factor, prefix in prefixes:
        if abs_val >= factor:
            return f"{value / factor:.2f} {prefix}{unit}".strip()
    return f"{value:.2e} {unit}".strip()

def clip_opamp(v_out, v_cc):
    """Apply realistic uA741 saturation limits (V_sat ≈ V_CC - 1.5 V)."""
    v_sat = max(v_cc - 1.5, 0.5)
    return np.clip(v_out, -v_sat, v_sat), v_sat
