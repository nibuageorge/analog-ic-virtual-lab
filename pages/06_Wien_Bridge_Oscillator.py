import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 6: Wien Bridge Oscillator", layout="wide")
render_sidebar_header()

st.title("Experiment 6: Wien Bridge Oscillator using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct a Wien bridge oscillator using **$\mu$A741** "
    "(Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Textbook Wien Bridge Oscillator Schematic
# -------------------------------------------------------------------------
def draw_wien_bridge_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.5)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Output node v_o
        d.add(elm.Line().right().at(op.out).length(0.75))
        out_node = d.add(elm.Dot())
        ox, oy = out_node.center
        d.add(elm.Line().right().at((ox, oy)).length(0.75).label('$v_o$', loc='right', fontsize=9.5))

        # =================================================================
        # UPPER BRANCH: Non-Inverting Amplifier (R1 to GND, Pot RF to Output)
        # =================================================================
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.15
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.45))
        d.add(elm.Line().up().toy(top_y))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # R1 going left to Ground
        d.add(elm.Resistor().left().at((sx, sy)).length(1.85).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='bottom', fontsize=8.5))
        d.add(elm.Line().down().length(0.35))
        d.add(elm.Ground())

        # Potentiometer RF (50-kΩ pot set at rf_kohm) connected as rheostat to Output
        d.add(elm.Potentiometer().right().at((sx, sy)).tox(ox).label(f'$R_F$ (50 kΩ pot)\nSet at {rf_kohm:.1f} kΩ', loc='top', fontsize=8))
        d.add(elm.Line().down().at((ox, sy)).toy(oy))

        # =================================================================
        # LOWER BRANCH: Wien Bridge Feedback Circuit (Series RC & Parallel RC)
        # =================================================================
        noninv_x, noninv_y = op.in2
        fb_left_x = sx - 2.35
        fb_bus_y = noninv_y - 1.65

        # Connection from Op-Amp Non-Inverting (+) terminal to Feedback Junction
        d.add(elm.Line().left().at((noninv_x, noninv_y)).tox(fb_left_x))
        d.add(elm.Line().down().toy(fb_bus_y))
        junc_a = d.add(elm.Dot())
        jx, jy = junc_a.center

        # Horizontal branch to right: Series R + Series C (labels cleanly on TOP)
        d.add(elm.Line().right().at((jx, jy)).length(1.15))
        junc_b = d.add(elm.Dot())
        bx, by = junc_b.center

        d.add(elm.Resistor().right().at((bx, by)).length(1.9).label(f'$R$\n{r_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        d.add(elm.Capacitor().right().tox(ox).label(f'$C$\n{c_uf:.3f} µF', loc='top', fontsize=8.5))
        d.add(elm.Line().up().toy(oy))

        # Parallel RC branch to Ground well below junc_b
        par_top_y = by - 0.85
        par_bot_y = par_top_y - 1.35
        d.add(elm.Line().down().at((bx, by)).toy(par_top_y))

        # Left leg of parallel network: Resistor R
        d.add(elm.Line().left().at((bx, par_top_y)).length(0.65))
        d.add(elm.Resistor().down().toy(par_bot_y).label(f'$R$\n{r_kohm:.1f} kΩ', loc='left', fontsize=8))
        d.add(elm.Line().right().length(0.65))

        # Right leg of parallel network: Capacitor C
        d.add(elm.Line().right().at((bx, par_top_y)).length(0.65))
        d.add(elm.Capacitor().down().toy(par_bot_y).label(f'$C$\n{c_uf:.3f} µF', loc='right', fontsize=8))
        d.add(elm.Line().left().length(0.65))

        # Common Ground below Parallel RC
        d.add(elm.Line().down().at((bx, par_bot_y)).length(0.35))
        d.add(elm.Ground())

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Time-Domain Simulation of Wien Bridge Oscillator (Rf = 2*R1)
# -------------------------------------------------------------------------
def simulate_wien_oscillator(R, C, R1, Rf, vcc, num_cycles=5, num_pts=3500):
    vsat = max(vcc - 1.5, 0.5)
    wo = 1.0 / (R * C)
    fo = wo / (2.0 * np.pi)

    # Non-inverting closed-loop gain: Av = 1 + Rf / R1 (Gayakwad: Rf = 2*R1 -> Av = 3)
    av = 1.0 + (Rf / R1)

    t_disp = np.linspace(0.0, num_cycles / fo, num_pts)

    if av < 2.98:
        # Below Barkhausen threshold (Rf < 2*R1): oscillations decay to 0 V
        decay_rate = (3.0 - av) * 0.35 * wo
        v_out_disp = 0.8 * np.exp(-decay_rate * t_disp) * np.sin(wo * t_disp)
        v_plus_disp = v_out_disp / 3.0
        thd_pct = 0.0
        return t_disp, v_plus_disp, v_out_disp, fo, 1.0 / fo, av, thd_pct, vsat

    if av <= 3.35:
        # Textbook Barkhausen design region (Rf = 2*R1 -> Av = 3.00): pure undistorted sine wave
        v_peak = vsat * (0.72 + 0.18 * ((av - 2.98) / 0.37))
        v_out_disp = v_peak * np.sin(wo * t_disp)
        v_plus_disp = v_out_disp / 3.0
        thd_pct = 0.10 + 0.65 * (abs(av - 3.0) / 0.35)
        return t_disp, v_plus_disp, v_out_disp, fo, 1.0 / fo, av, thd_pct, vsat

    # Over-driven region (Rf >> 2*R1): op-amp clips symmetrically at ±Vsat
    overdrive = 1.0 + 1.8 * (av - 3.35)
    raw_wave = overdrive * vsat * np.sin(wo * t_disp)
    v_out_disp = np.clip(raw_wave, -vsat, vsat)
    v_plus_disp = v_out_disp / av

    # Compute THD from FFT
    v_ac = v_out_disp - np.mean(v_out_disp)
    fft_vals = np.abs(np.fft.rfft(v_ac * np.hanning(len(v_ac))))
    fund_idx = np.argmax(fft_vals[1:]) + 1
    fund_amp = fft_vals[fund_idx]
    harmonics_sq = np.sum(fft_vals[2 * fund_idx::fund_idx] ** 2)
    thd_pct = (np.sqrt(harmonics_sq) / fund_amp) * 100.0 if fund_amp > 1e-6 else 0.0

    return t_disp, v_plus_disp, v_out_disp, fo, 1.0 / fo, av, thd_pct, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Feedback Network ($R, C$)")
    r_kohm = st.slider(
        "Wien Resistor $R$ (kΩ)",
        min_value=0.1,
        max_value=50.0,
        value=3.3,
        step=0.1,
        help="Default textbook value: 3.3 kΩ. Sets oscillation frequency fo = 1 / (2π R C)."
    )
    c_uf = st.slider(
        "Wien Capacitor $C$ (µF)",
        min_value=0.005,
        max_value=1.000,
        value=0.047,
        step=0.001,
        format="%.3f",
        help="Default textbook value: 0.05 µF."
    )

    st.subheader("2. Amplifier Gain Controls ($R_1, R_F$)")
    r1_kohm = st.slider(
        "Input Resistor $R_1$ (kΩ)",
        min_value=0.1,
        max_value=50.0,
        value=12.0,
        step=0.5,
        help="Default textbook value: 12 kΩ."
    )
    rf_kohm = st.slider(
        "Feedback Potentiometer $R_F$ (50-kΩ pot, set in kΩ)",
        min_value=0.1,
        max_value=50.0,
        value=24.0,
        step=0.5,
        help="Gayakwad design equation: RF = 2·R1 (for R1 = 12 kΩ, RF = 24 kΩ so Av = 1 + RF/R1 = 3.00)."
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=3, max_value=8, value=5, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R = r_kohm * 1e3
C = c_uf * 1e-6
R1 = r1_kohm * 1e3
Rf = rf_kohm * 1e3

t_disp, v_plus_disp, v_out_disp, fo, T, av, thd_pct, vsat = simulate_wien_oscillator(
    R, C, R1, Rf, vcc, num_cycles=num_cycles
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Closed-Loop Gain ($A_v$)", f"{av:.2f} (Req = 3.00)")
    m3.metric("Output $V_{o(pp)}$", f"{np.ptp(v_out_disp):.2f} V")
    m4.metric("Waveform THD", f"{thd_pct:.2f}%")

    if av < 2.98:
        st.error(
            f"**Oscillation Damps Out ($A_v = 1 + R_F/R_1 = {av:.2f} < 3.00$):** "
            f"Gayakwad requires **$R_F = 2 R_1 = {2*r1_kohm:.1f}\\text{{ k}}\\Omega$** ($A_v = 3.00$) to satisfy $A_v \\beta = 1$. "
            f"Increase potentiometer $R_F$ to **{2*r1_kohm:.1f} kΩ**."
        )
    elif av > 3.35:
        st.warning(
            f"**Over-Driven Gain ($A_v = {av:.2f} > 3.00$, $R_F > 2 R_1$):** "
            f"Setting $R_F$ significantly above $2 R_1 = {2*r1_kohm:.1f}\\text{{ k}}\\Omega$ clips the output at $\\pm V_{{sat}}$ (THD = {thd_pct:.1f}%). "
            f"Tune $R_F$ to **$2 R_1 = {2*r1_kohm:.1f}\\text{{ k}}\\Omega$** for a pure sine wave."
        )
    else:
        st.success(
            f"**Sustained Sinusoidal Oscillation ($f_o = {fo:.2f}\\text{{ Hz}}$, $R_F = {rf_kohm:.1f}\\text{{ k}}\\Omega \\approx 2 R_1$):** "
            f"Closed-loop gain $A_v = {av:.2f}$ balances the Wien bridge attenuation $\\beta = 1/3$ ($A_v \\beta = 1$) with pure sinusoidal output (THD = {thd_pct:.2f}%)."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad*)**")
        fig_sch = draw_wien_bridge_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Amplifier Gain:** $A_v = 1 + \\frac{{R_F}}{{R_1}} = 1 + \\frac{{{rf_kohm:.1f}}}{{{r1_kohm:.1f}}} = {av:.2f}$; "
            f"**Feedback Factor:** $\\beta = 1/3$ at $f_o = {fo:.1f}\\text{{ Hz}}$."
        )

    with cro_col:
        st.markdown("**Feedback ($v_+$) & Output ($v_o$) Waveforms (CRO)**")
        fig_cro, (ax_fb, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Non-inverting bridge feedback voltage v+(t)
        ax_fb.plot(t_ms, v_plus_disp, color="#1f77b4", linewidth=2.0, label="Bridge Feedback $v_+(t)$ (+ pin)")
        y_lim_fb = max(np.max(np.abs(v_plus_disp)) * 1.35, 0.5)
        ax_fb.set_ylim(-y_lim_fb, y_lim_fb)
        style_cro_axes(
            ax_fb,
            title=f"CH1: Wien Bridge Feedback $v_+(t)$ ($0^\\circ$ phase shift at $f_o$)",
            xlabel="",
            ylabel="$v_+$ (V)"
        )
        ax_fb.legend(loc="upper right", fontsize=8.5)

        # Channel 2: Sine Wave Output v_o(t)
        ax_out.plot(t_ms, v_out_disp, color="#d62728", linewidth=2.2, label="Oscillator Output $v_o(t)$")
        ax_out.axhline(vsat, color="#888888", linestyle=":", linewidth=1, label=f"$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V")
        ax_out.axhline(-vsat, color="#888888", linestyle=":", linewidth=1)
        ax_out.set_ylim(-vsat * 1.2, vsat * 1.2)
        style_cro_axes(
            ax_out,
            title=f"CH2: Wien Bridge Sine Output $v_o(t)$ ($f_o = {fo:.1f}$ Hz, $V_{{o(pp)}} = {np.ptp(v_out_disp):.2f}$ V)",
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
            "Select three Wien resistor values ($R_a, R_b, R_c$) with fixed $C = "
            f"{c_uf:.3f}\\text{{ µF}}$, $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$, and $R_F = {rf_kohm:.1f}\\text{{ k}}\\Omega$ "
            "to compare the sine-wave outputs across **three different frequencies**:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            ra_kohm = st.number_input("Trial 1: Resistor $R_a$ (kΩ)", min_value=1.0, max_value=50.0, value=1.5, step=0.1)
        with fcol2:
            rb_kohm = st.number_input("Trial 2: Resistor $R_b$ (kΩ)", min_value=1.0, max_value=50.0, value=3.3, step=0.1)
        with fcol3:
            rc_kohm = st.number_input("Trial 3: Resistor $R_c$ (kΩ)", min_value=1.0, max_value=50.0, value=6.8, step=0.1)

        trials = [
            ("Trial 1 ($f_1$)", ra_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", rb_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", rc_kohm * 1e3, "#d62728"),
        ]

        max_R = max(ra_kohm, rb_kohm, rc_kohm) * 1e3
        T_max_trial = 2.0 * np.pi * max_R * C
        t_common_span = 4.0 * T_max_trial

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r_val, color) in enumerate(trials):
            fo_i = 1.0 / (2.0 * np.pi * r_val * C)
            T_i = 1.0 / fo_i
            cycles_needed = max(4, int(np.ceil(t_common_span / T_i)))
            t_i, _, vout_i, _, _, _, thd_i, vsat_i = simulate_wien_oscillator(
                r_val, C, R1, Rf, vcc, num_cycles=cycles_needed, num_pts=4000
            )
            mask_i = t_i <= t_common_span
            ax = axes_cmp[idx]
            ax.plot(t_i[mask_i] * 1000.0, vout_i[mask_i], color=color, linewidth=2.0)
            ax.set_ylim(-vsat_i * 1.2, vsat_i * 1.2)
            style_cro_axes(
                ax,
                title=f"{label}: $R = {r_val/1e3:.2f}$ kΩ, $C = {c_uf:.3f}$ µF  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                xlabel="Time (ms)" if idx == 2 else "",
                ylabel="$v_o$ (V)"
            )
            table_rows.append({
                "Trial": label,
                "Wien Resistor R (kΩ)": f"{r_val/1e3:.2f}",
                "Wien Capacitor C (µF)": f"{c_uf:.3f}",
                "Time Period T (ms)": f"{T_i*1000:.3f}",
                "Frequency fo (Hz)": f"{fo_i:.2f}",
                "Output Vo(pp) (V)": f"{np.ptp(vout_i):.2f}",
                "THD (%)": f"{thd_i:.2f}%"
            })

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("**Observation Table (Three Frequencies)**")
        st.table(table_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Procedure (*R. A. Gayakwad, Section 7.4*)

            1. **Barkhausen Criterion & Frequency of Oscillation:**
               The Wien bridge oscillator consists of a non-inverting amplifier ($R_1, R_F$) and a lead-lag feedback network (series $RC$ and parallel $RC$). At the resonant frequency **$f_o$**, the phase shift of the feedback network is **$0^\\circ$** and the feedback fraction is **$\\beta = \\frac{{1}}{{3}}$**:
               $$f_o = \\frac{{1}}{{2\\pi R C}} = {fo:.2f}\\text{{ Hz}}$$

            2. **Closed-Loop Amplifier Gain ($A_v$):**
               To satisfy the Barkhausen criterion ($A_v \\beta = 1$), the non-inverting amplifier must provide a voltage gain of **$3$**:
               $$A_v = 1 + \\frac{{R_F}}{{R_1}} = 3 \\implies R_F = 2 R_1$$
               *(With $R_1 = 12\\text{{ k}}\\Omega$, the $50\\text{{ k}}\\Omega$ potentiometer $R_F$ is set at $24\\text{{ k}}\\Omega$ for sustained sinusoidal oscillation.)*

            3. **Design Steps for a Given Frequency $f_o$:**
               1. Choose a standard capacitor $C$ (e.g., $0.05\\text{{ µF}}$).
               2. Calculate $R = \\frac{{1}}{{2\\pi f_o C}}$.
               3. Select $R_1 = 12\\text{{ k}}\\Omega$ and use a $50\\text{{ k}}\\Omega$ potentiometer for $R_F$, setting $R_F = 2 R_1 = 24\\text{{ k}}\\Omega$ until a distortion-free sinusoidal output is observed on the CRO.
            """
        )
