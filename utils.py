import os
import streamlit as st
import numpy as np


def render_sidebar_header():
    for candidate in ["assets/logo.png", "logo.png", "assets/logo.jpg", "logo.jpg", "assets/Logo.png", "Logo.png"]:
        if os.path.exists(candidate):
            st.sidebar.image(candidate, use_container_width=True)
            break

    st.markdown(
        """
        <style>
        /* Main window padding */
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        /* Compact proportional headings across all pages */
        h1, [data-testid="stHeading"] h1 {
            font-size: 1.45rem !important;
            line-height: 1.25 !important;
        }
        h2, [data-testid="stHeading"] h2 {
            font-size: 1.15rem !important;
            line-height: 1.25 !important;
        }
        h3, [data-testid="stHeading"] h3 {
            font-size: 1.00rem !important;
            line-height: 1.25 !important;
            margin-top: 0.4rem !important;
            margin-bottom: 0.25rem !important;
        }
        /* Shrink st.metric numbers & prevent '...' truncation across all 4-column bars */
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] * {
            font-size: 1.05rem !important;
            line-height: 1.25 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetric"] [data-testid="stMetricLabel"] * {
            font-size: 0.78rem !important;
            line-height: 1.2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="
            background-color: #f0f4f8;
            color: #1e293b;
            padding: 12px 14px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            font-size: 0.85rem;
            line-height: 1.45;
            margin-top: 8px;
        ">
            This Virtual Lab is developed by<br>
            <strong>Dr. Nibu A George</strong>,<br>
            Associate Professor of Physics,<br>
            Baselius College (Autonomous),<br>
            Kottayam
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_cro_axes(ax, title="", xlabel="", ylabel=""):
    ax.axhline(0, color="#666666", linewidth=0.9, alpha=0.7)
    ax.axvline(0, color="#666666", linewidth=0.9, alpha=0.7)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")


def format_eng(val, unit=""):
    if val == 0:
        return f"0.00 {unit}".strip()
    abs_v = abs(val)
    prefixes = [
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
        (1e-12, "p"),
    ]
    for scale, sym in prefixes:
        if abs_v >= scale:
            return f"{val / scale:.2f} {sym}{unit}".strip()
    return f"{val:.2e} {unit}".strip()


def clip_opamp(v, vcc=15.0):
    vsat = max(vcc - 1.5, 0.5)
    return np.clip(v, -vsat, vsat), vsat
