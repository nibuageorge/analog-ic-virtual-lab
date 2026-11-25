import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 16: Asymmetric Astable Multivibrator (uA741)", layout="wide")
render_sidebar_header()

st.title("Experiment 16: Astable Multivibrator with Duty Cycle != 50% ($\\mu$A741)")
st.markdown(
    "Design and construct an **Astable Multivibrator (Square-Wave Generator)** using **$\\mu$A741** with steering diodes "
    "($D_1, D_2$) to obtain a **duty cycle other than 50%**, observe the output square wave $v_o(t)$ and capacitor voltage "
    "$v_C(t)$ on the CRO, and repeat the experiment for **two more duty cycle values** (*Gayakwad, Section 7-13*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Section 7-13 Asymmetric Astable Schematic
# -------------------------------------------------------------------------
def draw_asymmetric_astable_schematic(ra_kohm, rb_kohm, c_uf, r1_kohm, r2_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.0)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Output Node vo and Load Resistor RL ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center
        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o$', loc='right', fontsize=9.5))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # --- Upper Diode-Steered Timing Network (D1-RA & D2-RB) + Capacitor C ---
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.75))
        cap_node = d.add(elm.Dot().label('$v_C$', loc='left', fontsize=8.8))
        cx, cy = cap_node.center

        # Timing Capacitor C from (-) node to Ground
        d.add(elm.Capacitor().down().at((cx, cy)).length(1.65).label(f'$C$\n{c_uf:.3f} uF', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Upper Branch 1: Diode D1 (left->right or right->left? Charges C when vo = +Vsat -> D1 points left)
        b1_y = cy + 1.05
        d.add(elm.Line().up().at((cx, cy)).toy(b1_y))
        b1_tap = d.add(elm.Dot())
        d.add(elm.Resistor().right().at((cx, b1_y)).length(1.65).label(f'$R_A$\n{ra_kohm:.1f} k$\\Omega$', loc='top', fontsize=7.8))
        d.add(elm.Diode().right().reverse().tox(fb_x).label('$D_1$ (Charge)', loc='top', fontsize=7.5))
        d.add(elm.Line().down().toy(fb_y))

        # Upper Branch 2: Diode D2 (Discharges C when vo = -Vsat -> D2 points right)
        b2_y = b1_y + 1.05
        d.add(elm.Line().up().at(b1_tap.center).toy(b2_y))
        d.add(elm.Resistor().right().at((cx, b2_y)).length(1.65).label(f'$R_B$\n{rb_kohm:.1f} k$\\Omega$', loc='top', fontsize=7.8))
        d.add(elm.Diode().right().tox(fb_x).label('$D_2$ (Discharge)', loc='top', fontsize=7.5))
        d.add(elm.Line().down().toy(b1_y))

        # --- Lower Positive Feedback Divider (R1 to GND, R2 to Output vo) ---
        noninv_x, noninv_y = op.in2
        bot_y = noninv_y - 1.35
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.35))
        d.add(elm.Line().down().toy(bot_y))
        vref_node = d.add(elm.Dot().label('$V_{ref} = \\pm\\beta V_{sat}$', loc='top', fontsize=7.8))
        vx, vy = vref_node.center

        # R1 to Ground
        d.add(elm.Resistor().down().at((vx, vy)).length(1.2).label(f'$R_1$\n{r1_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # R2 to Output fb_tap
        d.add(elm.Resistor().right().at((vx, vy)).tox(fb_x).label(f'$R_2$\n{r2_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Line().up().toy(fb_y))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Helper: Exact Asymmetric Astable Timing & Waveform Generator
# -------------------------------------------------------------------------
def compute_astable_metrics(RA, RB, C, R1, R2, vcc, include_diode_drop=True):
    vsat = max(vcc - 1.0, 1.0)
    vd = 0.7 if include_diode_drop else 0.0
    beta = R1 / (R1 + R2)
    v_ut = beta * vsat
    v_lt = -beta * vsat
    v_drive = max(vsat - vd, v_ut + 0.2)

    ln_factor = np.log((v_drive + v_ut) / (v_drive - v_ut))
    t1 = RA * C * ln_factor  # Output HIGH (+Vsat) charging time
    t2 = RB * C * ln_factor  # Output LOW (-Vsat) discharging time
    period = t1 + t2
    freq = 1.0 / period
    duty_pct = (t1 / period) * 100.0
    return vsat, beta, v_ut, v_lt, v_drive, ln_factor, t1, t2, period, freq, duty_pct

def generate_astable_waveforms(t_arr, t1, t2, vsat, v_ut, v_lt, v_drive):
    period = t1 + t2
    t_mod = np.mod(t_arr, period)
    vo = np.where(t_mod < t1, vsat, -vsat)

    tau1 = t1 / np.log((v_drive + v_ut) / (v_drive - v_ut))
    tau2 = t2 / np.log((v_drive + v_ut) / (v_drive - v_ut))

    vc_charge = v_drive - (v_drive - v_lt) * np.exp(-t_mod / tau1)
    vc_discharge = -v_drive + (v_drive + v_ut) * np.exp(-(t_mod - t1) / tau2)
    vc = np.where(t_mod < t1, vc_charge, vc_discharge)
    return vo, vc

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Timing Network ($R_A, R_B, C$)")
    ra_kohm = st.slider(
        "Charging Resistor $R_A$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Controls output HIGH duration t1 (charging through D1)."
    )
    rb_kohm = st.slider(
        "Discharging Resistor $R_B$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=0.5,
        help="Controls output LOW duration t2 (discharging through D2). Default RA=10k, RB=30k -> Duty Cycle = 25%."
    )
    c_uf = st.slider(
        "Timing Capacitor $C$ (uF)",
        min_value=0.001,
        max_value=0.500,
        value=0.050,
        step=0.005,
        format="%.3f",
        help="Default 0.050 uF gives f0 ≈ 455 Hz for RA=10 kOhm, RB=30 kOhm."
    )

    st.subheader("2. Feedback Divider ($R_1, R_2, R_L$)")
    r1_kohm = st.slider(
        "Divider Resistor $R_1$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Sets feedback fraction beta = R1 / (R1 + R2) (default R1 = R2 = 10 kOhm -> beta = 0.50)."
    )
    r2_kohm = st.slider(
        "Divider Resistor $R_2$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )
    rl_kohm = st.slider(
        "Load Resistor $R_L$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    st.subheader("3. Diode Model & Supply")
    include_vd = st.checkbox("Include Silicon Diode Drop ($V_D = 0.7$ V)", value=True)
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Primary Astable Calculations
# -------------------------------------------------------------------------
RA = ra_kohm * 1e3
RB = rb_kohm * 1e3
C = c_uf * 1e-6
R1 = r1_kohm * 1e3
R2 = r2_kohm * 1e3

vsat, beta, v_ut, v_lt, v_drive, ln_fac, t1, t2, period, freq, duty_pct = compute_astable_metrics(
    RA, RB, C, R1, R2, vcc, include_vd
)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, CRO Waveforms & Repeat Tabs
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Freq ($f_o$)", format_eng(freq, "Hz"))
    m2.metric("Duty Cycle ($D$)", f"{duty_pct:.1f}% (T = {period*1e3:.2f} ms)")
    m3.metric("ON / OFF Times ($t_1, t_2$)", f"{t1*1e3:.2f} ms, {t2*1e3:.2f} ms")
    m4.metric("Thresholds ($\\pm \\beta V_{sat}$)", f"+/- {v_ut:.2f} V (beta={beta:.2f})")

    if np.isclose(duty_pct, 50.0, atol=0.5):
        st.info(
            f"**Symmetric 50% Duty Cycle ($R_A = R_B = {ra_kohm:.1f}$ kOhm):** "
            f"Change $R_A$ or $R_B$ so $R_A \\ne R_B$ to obtain an asymmetric duty cycle ($D \\ne 50\\%$). "
            f"Current frequency **$f_o = {freq:.1f}$ Hz**."
        )
    else:
        st.success(
            f"**Asymmetric Astable Multivibrator ($D = {duty_pct:.1f}\\% \\ne 50\\%$):** "
            f"HIGH time **$t_1 = {t1*1e3:.2f}$ ms** ($R_A = {ra_kohm:.1f}$ kOhm), "
            f"LOW time **$t_2 = {t2*1e3:.2f}$ ms** ($R_B = {rb_kohm:.1f}$ kOhm), "
            f"Total Period **$T = {period*1e3:.2f}$ ms**, Frequency **$f_o = {freq:.1f}$ Hz**."
        )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Sec. 7-13*)**")
        fig_sch = draw_asymmetric_astable_schematic(ra_kohm, rb_kohm, c_uf, r1_kohm, r2_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Formulas:** $t_1 = {ln_fac:.2f} R_A C$, $t_2 = {ln_fac:.2f} R_B C$, "
            f"$D = \\frac{{R_A}}{{R_A + R_B}} = {duty_pct:.1f}\\%$."
        )

    with cro_col:
        st.markdown("**CRO Waveforms: Output $v_o(t)$ & Capacitor $v_C(t)$**")
        t_arr = np.linspace(0.0, 3.0 * period, 2500)
        vo_arr, vc_arr = generate_astable_waveforms(t_arr, t1, t2, vsat, v_ut, v_lt, v_drive)
        t_ms = t_arr * 1000.0

        fig_cro, (ax_vo, ax_vc) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        # Top Trace: Output Square Wave vo(t)
        ax_vo.plot(t_ms, vo_arr, color="#d62728", linewidth=2.2, label=f"Output $v_o(t)$ (Duty Cycle = {duty_pct:.1f}%)")
        ax_vo.axhline(vsat, color="#888888", linestyle=":", linewidth=1.0, label=f"$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V")
        ax_vo.axhline(-vsat, color="#888888", linestyle=":", linewidth=1.0)
        ax_vo.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_vo,
            title=f"CH1: Asymmetric Square Wave $v_o(t)$ ($f_o = {freq:.1f}$ Hz, $D = {duty_pct:.1f}\\%$)",
            xlabel="",
            ylabel="$v_o$ (V)"
        )
        ax_vo.legend(loc="upper right", fontsize=8)

        # Bottom Trace: Exponential Capacitor Voltage vc(t)
        ax_vc.plot(t_ms, vc_arr, color="#1f77b4", linewidth=2.2, label="Capacitor Voltage $v_C(t)$")
        ax_vc.axhline(v_ut, color="#2ca02c", linestyle="--", linewidth=1.1, label=f"$+V_{{UT}} = +{v_ut:.2f}$ V")
        ax_vc.axhline(v_lt, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$-V_{{LT}} = {v_lt:.2f}$ V")
        ax_vc.set_ylim(-vsat * 1.15, vsat * 1.15)
        style_cro_axes(
            ax_vc,
            title=f"CH2: Capacitor Charging/Discharging $v_C(t)$ ($t_1 = {t1*1e3:.2f}$ ms, $t_2 = {t2*1e3:.2f}$ ms)",
            xlabel="Time (ms)",
            ylabel="$v_C$ (V)"
        )
        ax_vc.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Repeat for Two More Duty Cycle Values + Theory
    # ---------------------------------------------------------------------
    tab_repeat, tab_theory = st.tabs([
        "Repeat for Two More Duty Cycle Values (3-Trial Comparison)",
        "Theory & Design Equations (Gayakwad Sec. 7-13)"
    ])

    with tab_repeat:
        st.markdown(
            f"Compare **Trial 1** (current setting: **$D_1 = {duty_pct:.1f}\\%$**, $R_A = {ra_kohm:.1f}$ kOhm, $R_B = {rb_kohm:.1f}$ kOhm) "
            f"with **two more duty cycle values** ($D_2$ and $D_3$) at the **same total resistance $R_A + R_B = {ra_kohm + rb_kohm:.1f}$ kOhm** "
            f"(keeping frequency **$f_o = {freq:.1f}$ Hz** constant while varying $D = \\frac{{R_A}}{{R_A + R_B}}$):"
        )

        c_d2, c_d3 = st.columns(2, gap="medium")
        with c_d2:
            d2_target = st.slider("Trial 2 Duty Cycle $D_2$ (%)", min_value=10, max_value=90, value=60, step=5)
        with c_d3:
            d3_target = st.slider("Trial 3 Duty Cycle $D_3$ (%)", min_value=10, max_value=90, value=80, step=5)

        R_sum = RA + RB
        RA_2 = (d2_target / 100.0) * R_sum
        RB_2 = R_sum - RA_2
        RA_3 = (d3_target / 100.0) * R_sum
        RB_3 = R_sum - RA_3

        _, _, _, _, _, _, t1_2, t2_2, T_2, f_2, D_2 = compute_astable_metrics(RA_2, RB_2, C, R1, R2, vcc, include_vd)
        _, _, _, _, _, _, t1_3, t2_3, T_3, f_3, D_3 = compute_astable_metrics(RA_3, RB_3, C, R1, R2, vcc, include_vd)

        t_cmp = np.linspace(0.0, 2.5 * period, 2200)
        t_cmp_ms = t_cmp * 1000.0
        vo_1, vc_1 = generate_astable_waveforms(t_cmp, t1, t2, vsat, v_ut, v_lt, v_drive)
        vo_2, vc_2 = generate_astable_waveforms(t_cmp, t1_2, t2_2, vsat, v_ut, v_lt, v_drive)
        vo_3, vc_3 = generate_astable_waveforms(t_cmp, t1_3, t2_3, vsat, v_ut, v_lt, v_drive)

        fig_rep, axes_rep = plt.subplots(3, 1, figsize=(8.8, 5.8), sharex=True)
        trials_plot = [
            (axes_rep[0], vo_1, vc_1, f"Trial 1: D1 = {duty_pct:.1f}% (RA={ra_kohm:.1f} kOhm, RB={rb_kohm:.1f} kOhm, t1={t1*1e3:.2f} ms, t2={t2*1e3:.2f} ms)", "#d62728"),
            (axes_rep[1], vo_2, vc_2, f"Trial 2: D2 = {D_2:.1f}% (RA={RA_2/1e3:.1f} kOhm, RB={RB_2/1e3:.1f} kOhm, t1={t1_2*1e3:.2f} ms, t2={t2_2*1e3:.2f} ms)", "#1f77b4"),
            (axes_rep[2], vo_3, vc_3, f"Trial 3: D3 = {D_3:.1f}% (RA={RA_3/1e3:.1f} kOhm, RB={RB_3/1e3:.1f} kOhm, t1={t1_3*1e3:.2f} ms, t2={t2_3*1e3:.2f} ms)", "#2ca02c"),
        ]
        for idx_p, (ax_p, vo_p, vc_p, title_p, col_p) in enumerate(trials_plot):
            ax_p.plot(t_cmp_ms, vo_p, color=col_p, linewidth=2.0, label="Output $v_o(t)$")
            ax_p.plot(t_cmp_ms, vc_p, color="#555555", linestyle="--", linewidth=1.3, label="Capacitor $v_C(t)$")
            ax_p.set_ylim(-vsat * 1.25, vsat * 1.25)
            style_cro_axes(
                ax_p,
                title=title_p,
                xlabel="Time (ms)" if idx_p == 2 else "",
                ylabel="V"
            )
            ax_p.legend(loc="upper right", fontsize=7.5)

        plt.tight_layout()
        st.pyplot(fig_rep)
        plt.close(fig_rep)

        st.markdown("**Observation Table for Three Duty Cycle Values**")
        st.table([
            {
                "Trial": "Trial 1 (Primary)",
                "RA (kOhm)": f"{ra_kohm:.2f}",
                "RB (kOhm)": f"{rb_kohm:.2f}",
                "C (uF)": f"{c_uf:.3f}",
                "HIGH Time t1 (ms)": f"{t1*1e3:.2f}",
                "LOW Time t2 (ms)": f"{t2*1e3:.2f}",
                "Period T (ms)": f"{period*1e3:.2f}",
                "Frequency fo (Hz)": f"{freq:.1f}",
                "Duty Cycle D (%)": f"{duty_pct:.1f}%"
            },
            {
                "Trial": "Trial 2 (Repeat 1)",
                "RA (kOhm)": f"{RA_2/1e3:.2f}",
                "RB (kOhm)": f"{RB_2/1e3:.2f}",
                "C (uF)": f"{c_uf:.3f}",
                "HIGH Time t1 (ms)": f"{t1_2*1e3:.2f}",
                "LOW Time t2 (ms)": f"{t2_2*1e3:.2f}",
                "Period T (ms)": f"{T_2*1e3:.2f}",
                "Frequency fo (Hz)": f"{f_2:.1f}",
                "Duty Cycle D (%)": f"{D_2:.1f}%"
            },
            {
                "Trial": "Trial 3 (Repeat 2)",
                "RA (kOhm)": f"{RA_3/1e3:.2f}",
                "RB (kOhm)": f"{RB_3/1e3:.2f}",
                "C (uF)": f"{c_uf:.3f}",
                "HIGH Time t1 (ms)": f"{t1_3*1e3:.2f}",
                "LOW Time t2 (ms)": f"{t2_3*1e3:.2f}",
                "Period T (ms)": f"{T_3*1e3:.2f}",
                "Frequency fo (Hz)": f"{f_3:.1f}",
                "Duty Cycle D (%)": f"{D_3:.1f}%"
            }
        ])

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7-13*)

            1. **Principle of Asymmetric Duty Cycle Operation:**
               * In a standard op-amp astable multivibrator with a single feedback resistor $R_F$, the charging and discharging time constants are equal ($t_1 = t_2$), fixing the duty cycle at $50\\%$.
               * By splitting the feedback path into two parallel branches with steering diodes **$D_1$** and **$D_2$**:
                 * When $v_o = +V_{{sat}}$, diode **$D_1$** is forward-biased and **$D_2$** is reverse-biased, so capacitor $C$ charges from $-\\beta V_{{sat}}$ to $+\\beta V_{{sat}}$ solely through **$R_A$**.
                 * When $v_o = -V_{{sat}}$, diode **$D_2$** is forward-biased and **$D_1$** is reverse-biased, so capacitor $C$ discharges from $+\\beta V_{{sat}}$ to $-\\beta V_{{sat}}$ solely through **$R_B$**.

            2. **Timing & Duty Cycle Equations:**
               * **Feedback Fraction:** $\\beta = \\frac{{R_1}}{{R_1 + R_2}} = {beta:.3f}$ (Upper/Lower Thresholds: $\\pm V_{{ref}} = \\pm \\beta V_{{sat}} = \\pm {v_ut:.2f}\\text{{ V}}$)
               * **Output HIGH Time ($t_1$):**
                 $$t_1 = R_A C \\ln\\left(\\frac{{V_{{sat}} - V_D + \\beta V_{{sat}}}}{{V_{{sat}} - V_D - \\beta V_{{sat}}}}\\right) = {t1*1e3:.2f}\\text{{ ms}}$$
               * **Output LOW Time ($t_2$):**
                 $$t_2 = R_B C \\ln\\left(\\frac{{V_{{sat}} - V_D + \\beta V_{{sat}}}}{{V_{{sat}} - V_D - \\beta V_{{sat}}}}\\right) = {t2*1e3:.2f}\\text{{ ms}}$$
               * **Total Time Period ($T$), Frequency ($f_o$), and Duty Cycle ($D$):**
                 $$T = t_1 + t_2 = {period*1e3:.2f}\\text{{ ms}}, \\qquad f_o = \\frac{{1}}{{T}} = {freq:.1f}\\text{{ Hz}}, \\qquad D = \\frac{{t_1}}{{t_1 + t_2}} \\times 100\\% = \\frac{{R_A}}{{R_A + R_B}} \\times 100\\% = {duty_pct:.1f}\\%$$
            """
        )