import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render_sidebar_header():
    """Display developer and institutional attribution at the top of the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="background-color: #f0f4f8; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 12px;">
                <p style="margin: 0; font-size: 0.88rem; color: #1e293b; line-height: 1.45;">
                    This Virtual Lab is developed by<br>
                    <strong>Dr. Nibu A George</strong>,<br>
                    Associate Professor of Physics,<br>
                    Baselius College (Autonomous), Kottayam
                </p>
            </div>
            """,
            unsafe_allow_html=True
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
