import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 3: Square Wave Generator", layout="wide")
render_sidebar_header()

st.title("Experiment 3: Square Wave Generator using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct a free-running square wave generator (astable multivibrator) using **$\mu$A741** "
    "for a target frequency $f_o$, plot the capacitor and output waveforms, and repeat the experiment for "
    "two additional frequencies (*Gayakwad, Section 7.6*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Square-Wave Generator Schematic (Fig. 7-13)
# -------------------------------------------------------------------------
def draw_square_gen_schematic(r_kohm, c_uf, r1_kohm, r2_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=9.2)

        # Op-Amp with non-inverting (+) at top, inverting (-) at bottom, or standard layout
        # Using standard layout: in1 (-) top, in2 (+) bottom
        op = d.add(elm.Opamp(leads=True).label('$\mu$A741', loc='center', fontsize=8.5))

        # Power supply pins
        d.add(elm.Line().up().at(op.vd).length(0.45).label(f'+{vcc:.0f}V (7)', loc='top', fontsize=8))
        d.add(elm.Line().down().at(op.vs).length(0.45).label(f'-{vcc:.0f}V (4)', loc='bottom', fontsize=8))

        # Output node
        d.add(elm.Line().right().at(op.out).length(0.8))
        out_node = d.add(elm.Dot())
        d.add(elm.Line().right().length(0.9).label('$v_o$ (6)', loc='right'))

        # --- Upper Feedback Loop (Negative RC timing branch to Pin 2) ---
        d.add(elm.Line().left().at(op.in1).length(0.6))
        inv_node = d.add(elm.Dot())
        d.add(elm.Capacitor().down().length(1.3).label(f'$C$\n{c_uf:.3f} µF', loc='bottom'))
        d.add(elm.Ground())

        d.add(elm.Line().up().at(inv_node.center).length(1.35))
        d.add(elm.Resistor().right().tox(out_node.center).label(f'$R$\n{r_kohm:.1f} kΩ', loc='top'))
        d.add(elm.Line().down().toy(out_node.center))

        # --- Lower Feedback Loop (Positive voltage divider R1, R2 to Pin 3) ---
        d.add(elm.Line().left().at(op.in2).length(0.6))
        d.add(elm.Line().down().length(1.25))
        div_node = d.add(elm.Dot())

        # R1 to Ground
        d.add(elm.Resistor().down().length(1.25).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Ground())

        # R2 from divider node to Output
        d.add(elm.Resistor().right().at(div_node.center).tox(out_node.center).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Line().up().toy(out_node.center))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Generate Exact Analytical Waveforms for Square Wave Generator
# -------------------------------------------------------------------------
def simulate_square_generator(R, C, R1, R2, vcc, t_span, num_pts=3000):
    vsat = max(vcc - 1.5, 0.5)
    beta = R1 / (R1 + R2)
    v_ut = beta * vsat
    v_lt = -beta * vsat
    tau = R * C

    # Half-period T/2 = R * C * ln((1 + beta) / (1 - beta))
    half_T = tau * np.log((1.0 + beta) / (1.0 - beta))
    T = 2.0 * half_T
    fo = 1.0 / T

    t = np.linspace(0.0, t_span, num_pts)
    t_mod = np.mod(t, T)

    v_out = np.empty_like(t)
    v_cap = np.empty_like(t)

    # First half-cycle [0, T/2): v_out = +Vsat, capacitor charges from -beta*Vsat toward +Vsat
    mask1 = t_mod < half_T
    t1 = t_mod[mask1]
    v_out[mask1] = vsat
    v_cap[mask1] = vsat - (vsat - v_lt) * np.exp(-t1 / tau)

    # Second half-cycle [T/2, T): v_out = -Vsat, capacitor discharges from +beta*Vsat toward -Vsat
    mask2 = ~mask1
    t2 = t_mod[mask2] - half_T
    v_out[mask2] = -vsat
    v_cap[mask2] = -vsat + (v_ut - (-vsat)) * np.exp(-t2 / tau)

    return t, v_cap, v_out, fo, T, beta, v_ut, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Timing Components ($R, C$)")
    r_kohm = st.slider(
        "Timing Resistor $R$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Varying R directly changes the oscillation frequency fo."
    )
    c_uf = st.slider(
        "Timing Capacitor $C$ (µF)",
        min_value=0.005,
        max_value=0.500,
        value=0.050,
        step=0.005,
        format="%.3f"
    )

    st.subheader("2. Positive Feedback Divider ($R_1, R_2$)")
    preset_beta = st.checkbox(
        "Use Gayakwad Simplification ($R_2 = 1.16 R_1 \\Rightarrow f_o = 1/2RC$)",
        value=True,
        help="When R2 = 1.16·R1, ln((2R1 + R2)/R2) = 1.00, simplifying fo to 1/(2RC)."
    )
    r1_kohm = st.slider("Feedback Resistor $R_1$ (kΩ)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
    if preset_beta:
        r2_kohm = round(1.16 * r1_kohm, 2)
        st.caption(f"Matched Resistor: **$R_2 = 1.16 R_1 = {r2_kohm:.2f}\\text{{ k}}\\Omega$**")
    else:
        r2_kohm = st.slider("Feedback Resistor $R_2$ (kΩ)", min_value=1.0, max_value=50.0, value=11.6, step=0.5)

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=2, max_value=8, value=4, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R = r_kohm * 1e3
C = c_uf * 1e-6
R1 = r1_kohm * 1e3
R2 = r2_kohm * 1e3

# Preliminary period calculation to set display window
beta_tmp = R1 / (R1 + R2)
T_tmp = 2.0 * R * C * np.log((1.0 + beta_tmp) / (1.0 - beta_tmp))
t_disp, v_cap, v_out, fo, T, beta, v_ut, vsat = simulate_square_generator(
    R, C, R1, R2, vcc, t_span=num_cycles * T_tmp
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Time Period ($T$)", format_eng(T, "s"))
    m3.metric("Thresholds ($\\pm V_{{UT}}$)", f"±{v_ut:.2f} V (β={beta:.3f})")
    m4.metric("Output Amplitude ($V_{{o(pp)}}$)", f"{2*vsat:.1f} V")

    # Slew rate check for uA741 (SR = 0.5 V/us)
    rise_time_us = (2.0 * vsat) / 0.5
    if fo > 15000:
        st.warning(
            f"**Slew-Rate Limitation Notice ($f_o = {format_eng(fo, 'Hz')}$):** "
            f"The $\\mu$A741 has a typical slew rate of $0.5\\text{{ V/µs}}$ (requiring $\\approx {rise_time_us:.0f}\\text{{ µs}}$ to swing from $-V_{{sat}}$ to $+V_{{sat}}$). "
            f"At high frequencies, square-wave edges appear trapezoidal on a real CRO."
        )
    else:
        st.success(
            f"**Self-Sustaining Square Wave Oscillation ($f_o = {fo:.2f}\\text{{ Hz}}$, Duty Cycle = $50\\%$):** "
            f"Capacitor $C$ charges and discharges exponentially between $V_{{LT}} = -{v_ut:.2f}\\text{{ V}}$ and $V_{{UT}} = +{v_ut:.2f}\\text{{ V}}$."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-13*)**")
        fig_sch = draw_square_gen_schematic(r_kohm, c_uf, r1_kohm, r2_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**$\mu$A741 Pinout:** Pin 2 = Inverting (-), Pin 3 = Non-inverting (+), "
            f"Pin 4 = $-V_{{CC}}$, Pin 6 = Output ($v_o$), Pin 7 = $+V_{{CC}}$."
        )

    with cro_col:
        st.markdown("**Capacitor & Output Waveforms (CRO)**")
        fig_cro, (ax_cap, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Capacitor Voltage v_C(t)
        ax_cap.plot(t_ms, v_cap, color="#1f77b4", linewidth=2.1, label="Capacitor Voltage $v_C(t)$ (Pin 2)")
        ax_cap.axhline(v_ut, color="#2ca02c", linestyle="--", linewidth=1.1, label=f"$+V_{{UT}} = +{v_ut:.2f}$ V")
        ax_cap.axhline(-v_ut, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$V_{{LT}} = -{v_ut:.2f}$ V")
        ax_cap.set_ylim(-vsat * 1.15, vsat * 1.15)
        style_cro_axes(
            ax_cap,
            title=f"CH1: Capacitor Voltage $v_C(t)$ (Exponential Ramp between $\\pm\\beta V_{{sat}}$)",
            xlabel="",
            ylabel="$v_C$ (V)"
        )
        ax_cap.legend(loc="upper right", fontsize=8, ncol=2)

        # Channel 2: Square Wave Output v_o(t)
        ax_out.plot(t_ms, v_out, color="#d62728", linewidth=2.2, label="Square Wave Output $v_o(t)$ (Pin 6)")
        ax_out.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_out,
            title=f"CH2: Square Wave Output ($f_o = {fo:.1f}$ Hz, $T = {T*1000:.2f}$ ms)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Multi-Frequency Comparison (3 Frequencies) + Theory
    # ---------------------------------------------------------------------
    tab_multi, tab_theory = st.tabs([
        "🔁 Repeat for Three Frequencies ($f_1, f_2, f_3$)",
        "📖 Theory & Design Procedure (Gayakwad)"
    ])

    with tab_multi:
        st.markdown(
            "Select three timing resistor values ($R_a, R_b, R_c$) with fixed $C = "
            f"{c_uf:.3f}\\text{{ µF}}$, $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$, and $R_2 = {r2_kohm:.2f}\\text{{ k}}\\Omega$ "
            "to compare the square-wave outputs across **three different frequencies** simultaneously:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            ra_kohm = st.number_input("Trial 1: Resistor $R_a$ (kΩ)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
        with fcol2:
            rb_kohm = st.number_input("Trial 2: Resistor $R_b$ (kΩ)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        with fcol3:
            rc_kohm = st.number_input("Trial 3: Resistor $R_c$ (kΩ)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

        trials = [
            ("Trial 1 ($f_1$)", ra_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", rb_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", rc_kohm * 1e3, "#d62728"),
        ]

        # Common time axis covering 4 cycles of the lowest-frequency (longest-period) trial
        max_R = max(ra_kohm, rb_kohm, rc_kohm) * 1e3
        t_common_span = 4.0 * (2.0 * max_R * C * np.log((1.0 + beta) / (1.0 - beta)))

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r_val, color) in enumerate(trials):
            t_i, _, vout_i, fo_i, T_i, _, _, vsat_i = simulate_square_generator(
                r_val, C, R1, R2, vcc, t_span=t_common_span, num_pts=3000
            )
            ax = axes_cmp[idx]
            ax.plot(t_i * 1000.0, vout_i, color=color, linewidth=2.0)
            ax.set_ylim(-vsat_i * 1.25, vsat_i * 1.25)
            style_cro_axes(
                ax,
                title=f"{label}: $R = {r_val/1e3:.1f}$ kΩ, $C = {c_uf:.3f}$ µF  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                xlabel="Time (ms)" if idx == 2 else "",
                ylabel="$v_o$ (V)"
            )
            table_rows.append({
                "Trial": label,
                "Timing Resistor R (kΩ)": f"{r_val/1e3:.1f}",
                "Timing Capacitor C (µF)": f"{c_uf:.3f}",
                "Time Period T (ms)": f"{T_i*1000:.3f}",
                "Frequency fo (Hz)": f"{fo_i:.2f}",
                "Output Vo(pp) (V)": f"{2*vsat_i:.1f}"
            })

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("**Observation Table (Three Frequencies)**")
        st.table(table_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7.6*)

            1. **Principle of Operation (Astable Multivibrator):**
               The $\\mu$A741 operates in saturation ($\\pm V_{{sat}} \\approx \\pm {vsat:.1f}\\text{{ V}}$) due to positive feedback through voltage divider $R_1, R_2$, with feedback fraction:
               $$\\beta = \\frac{{R_1}}{{R_1 + R_2}} = {beta:.3f}$$
               * When $v_o = +V_{{sat}}$, the non-inverting threshold is the **Upper Threshold Point**:
                 $$V_{{UT}} = +\\beta V_{{sat}} = +{v_ut:.2f}\\text{{ V}}$$
               * Capacitor $C$ charges exponentially toward $+V_{{sat}}$ through resistor $R$. The instant $v_C$ crosses $+V_{{UT}}$, the op-amp switches to $v_o = -V_{{sat}}$, shifting the threshold to the **Lower Threshold Point**:
                 $$V_{{LT}} = -\\beta V_{{sat}} = -{v_ut:.2f}\\text{{ V}}$$
               * Capacitor $C$ then discharges toward $-V_{{sat}}$ until $v_C$ crosses $V_{{LT}}$, repeating the cycle continuously.

            2. **Time Period & Frequency of Oscillation:**
               $$T = 2 R C \\ln\\left(\\frac{{1 + \\beta}}{{1 - \\beta}}\\right) = 2 R C \\ln\\left(\\frac{{2 R_1 + R_2}}{{R_2}}\\right) = {T*1000:.3f}\\text{{ ms}}$$
               $$f_o = \\frac{{1}}{{T}} = \\frac{{1}}{{2 R C \\ln\\left(\\frac{{2 R_1 + R_2}}{{R_2}}\\right)}} = {fo:.2f}\\text{{ Hz}}$$

            3. **Gayakwad Simplified Design Rule:**
               If we choose **$R_2 = 1.16 R_1$** (so $\\beta = 0.463$), then $\\ln\\left(\\frac{{2R_1 + R_2}}{{R_2}}\\right) = \\ln(2.724) \\approx 1.00$, which simplifies the design equation to:
               $$f_o = \\frac{{1}}{{2 R C}}$$
               **Design Steps for a Given Frequency $f_o$:**
               1. Choose $R_1$ (typically $10\\text{{ k}}\\Omega$) and set $R_2 = 1.16 R_1 = 11.6\\text{{ k}}\\Omega$.
               2. Assume a standard capacitor $C$ (e.g., $0.05\\text{{ µF}}$ or $0.1\\text{{ µF}}$).
               3. Calculate the timing resistor $R = \\frac{{1}}{{2 f_o C}}$.
            """
        )