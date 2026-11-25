import streamlit as st
import numpy as np


def render_sidebar_header():
    st.markdown(
        """
        <style>
        /* Main window layout & typography */
        section[data-testid="stMain"] .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section[data-testid="stMain"] h1 {
            font-size: 1.5rem !important;
            line-height: 1.25 !important;
        }
        section[data-testid="stMain"] h2 {
            font-size: 1.15rem !important;
        }
        section[data-testid="stMain"] h3 {
            font-size: 1.0rem !important;
        }
        /* Shrink main-window st.metric numbers & prevent '...' truncation */
        section[data-testid="stMain"] [data-testid="stMetricValue"],
        section[data-testid="stMain"] [data-testid="stMetricValue"] * {
            font-size: 1.05rem !important;
            line-height: 1.25 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }
        section[data-testid="stMain"] [data-testid="stMetricLabel"],
        section[data-testid="stMain"] [data-testid="stMetricLabel"] * {
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
