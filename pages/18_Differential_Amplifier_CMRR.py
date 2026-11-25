import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 20: Differential Amplifier & CMRR", layout="wide")
render_sidebar_header()

st.title("Experiment 20: Differential Amplifier & CMRR Measurement ($\\mu$A741)")
st.markdown(
    "Design and construct a **Differential Amplifier with One Op-Amp** using **$\\mu$A741**, "
    "measure output voltages in **Differential Mode** and **Common Mode** for various input voltages at a fixed frequency, "
    "and determine the **Common-Mode Rejection Ratio (CMRR)** (*Gayakwad, Section 3-5-1, Figure 3-14, Eqs. 3-24 & 3-25*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 3-14 Differential Amplifier Schematic
# -------------------------------------------------------------------------
def draw_diff_amp_schematic(r1_kohm, r2_kohm, rf_kohm, r3_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        # --- Op-Amp (741): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('A', loc='center', fontsize=9.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+$V_{{CC}}$ (+{vcc:.0f}V)', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-$V_{{EE}}$ (-{vcc:.0f}V)', loc='bottom', fontsize=7.5))

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

        # --- Upper Inverting Branch: vx -> R1 -> node v2 -> RF -> Output ---
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.15
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.55))
        d.add(elm.Line().up().toy(top_y))
        v2_node = d.add(elm.Dot().label('$v_2$', loc='top', fontsize=8.5))
        v2_x, v2_y = v2_node.center

        # Feedback Resistor RF to Output fb_tap
        d.add(elm.Resistor().right().at((v2_x, v2_y)).tox(fb_x).label(f'$R_F$\n{rf_kohm:.2f} k$\\Omega$', loc='top', fontsize=8.2))
        d.add(elm.Line().down().toy(fb_y))

        # Input Resistor R1 from Source vx to node v2
        d.add(elm.Resistor().left().at((v2_x, v2_y)).length(1.95).label(f'$R_1$\n{r1_kohm:.2f} k$\\Omega$', loc='top', fontsize=8.2))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, v2_y)).length(0.2))
        d.add(elm.SourceSin().down().length(1.05).label('$v_x$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # --- Lower Non-Inverting Branch: vy -> R2 -> node v1 -> R3 -> Ground ---
        noninv_x, noninv_y = op.in2
        bot_y = noninv_y - 0.55
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.55))
        d.add(elm.Line().down().toy(bot_y))
        v1_node = d.add(elm.Dot().label('$v_1$', loc='top', fontsize=8.5))
        v1_x, v1_y = v1_node.center

        # Divider Resistor R3 from node v1 to Ground
        d.add(elm.Resistor().down().at((v1_x, v1_y)).length(1.35).label(f'$R_3$\n{r3_kohm:.2f} k$\\Omega$', loc='right', fontsize=8.2))
        d.add(elm.Ground())

        # Input Resistor R2 from Source vy to node v1
        d.add(elm.Resistor().left().at((v1_x, v1_y)).length(1.95).label(f'$R_2$\n{r2_kohm:.2f} k$\\Omega$', loc='top', fontsize=8.2))
        src_y_x = d.here[0]
        d.add(elm.Line().down().at((src_y_x, v1_y)).length(0.2))
        d.add(elm.SourceSin().down().length(1.05).label('$v_y$', loc='left', fontsize=9))
        d.add(elm.Ground())

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Helper: Exact Differential Amplifier Gains & CMRR Calculation
# -------------------------------------------------------------------------
def compute_diff_amp_gains(R1, R2, RF, R3, internal_cmrr_db):
    # Gayakwad Section 3-5-1(c) exact superposition gains:
    # vo = -(RF/R1)*vx + [R3/(R2 + R3)] * (1 + RF/R1) * vy
    gain_x = -(RF / R1)
    gain_y = (R3 / (R2 + R3)) * (1.0 + (RF / R1))

    # Differential gain AD = vo / vxy where vxy = vx - vy (with vx = +vxy/2, vy = -vxy/2)
    ad = 0.5 * (gain_x - gain_y)

    # Common-mode gain ACM = vo / vcm (when vx = vy = vcm)
    # Includes resistor-mismatch term (gain_x + gain_y) plus finite op-amp internal CMRR contribution
    acm_mismatch = gain_x + gain_y
    acm_opamp = abs(ad) / (10.0 ** (internal_cmrr_db / 20.0))
    acm_total = abs(acm_mismatch) + acm_opamp

    cmrr_ratio = abs(ad) / max(acm_total, 1e-12)
    cmrr_db = 20.0 * np.log10(cmrr_ratio)
    return gain_x, gain_y, ad, acm_mismatch, acm_total, cmrr_ratio, cmrr_db

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Waveforms)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Bridge Resistors ($R_1 = R_2, R_F = R_3$)")
    r1_kohm = st.slider(
        "Inverting Input Resistor $R_1$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
        help="Gayakwad Fig. 3-14 requires R1 = R2 (default 10.0 kOhm)."
    )
    r2_kohm = st.slider(
        "Non-Inverting Input Resistor $R_2$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
        help="Slightly varying R2 relative to R1 demonstrates resistor-mismatch degradation of CMRR."
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kOhm)",
        min_value=1.0,
        max_value=200.0,
        value=100.0,
        step=1.0,
        help="Sets differential gain AD = -RF / R1 (default 100 kOhm with R1 = 10 kOhm -> |AD| = 10)."
    )
    r3_kohm = st.slider(
        "Ground Divider Resistor $R_3$ (kOhm)",
        min_value=1.0,
        max_value=200.0,
        value=100.0,
        step=1.0,
        help="Gayakwad Fig. 3-14 requires R3 = RF (default 100.0 kOhm)."
    )
    rl_kohm = st.slider("Load Resistor $R_L$ (kOhm)", 1.0, 50.0, 10.0, 1.0)

    st.subheader("2. Operating Mode & Input Signals")
    op_mode = st.radio(
        "Test Mode Selection",
        ["Differential Mode (vxy = vx - vy)", "Common Mode (vx = vy = vcm)"],
        index=0
    )
    vin_test_pp = st.slider(
        "Test Input Amplitude $V_{in(pp)}$ (V)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Differential input vxy(pp) in Differential Mode, or common-mode input vcm(pp) in Common Mode."
    )
    freq_hz = st.slider("Signal Frequency $f$ (Hz)", 100, 5000, 1000, 100)

    st.subheader("3. Op-Amp Internal CMRR & Supply")
    internal_cmrr_db = st.slider(
        "uA741 Internal CMRR (dB)",
        min_value=60,
        max_value=110,
        value=90,
        step=5,
        help="Typical uA741 datasheet CMRR is 90 dB."
    )
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Circuit & CMRR Calculations (Gayakwad Section 3-5-1, Eqs. 3-24 & 3-25)
# -------------------------------------------------------------------------
R1 = r1_kohm * 1e3
R2 = r2_kohm * 1e3
RF = rf_kohm * 1e3
R3 = r3_kohm * 1e3
vsat = max(vcc - 1.0, 0.5)

gain_x, gain_y, ad, acm_mismatch, acm_total, cmrr_ratio, cmrr_db = compute_diff_amp_gains(
    R1, R2, RF, R3, internal_cmrr_db
)

vo_diff_pp = min(abs(ad) * vin_test_pp, 2.0 * vsat)
vo_cm_pp = min(acm_total * vin_test_pp, 2.0 * vsat)
balanced = np.isclose(r1_kohm, r2_kohm, atol=1e-4) and np.isclose(rf_kohm, r3_kohm, atol=1e-4)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, CRO Waveforms & Observation Tables
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Differential Gain ($|A_D|$)", f"{abs(ad):.2f} ({20*np.log10(max(abs(ad), 1e-6)):+.1f} dB)")
    m2.metric("Common-Mode Gain ($|A_{CM}|$)", f"{acm_total:.2e}")
    m3.metric("CMRR (Linear Ratio)", f"{cmrr_ratio:,.0f}")
    m4.metric("CMRR (dB)", f"{cmrr_db:.2f} dB")

    if not balanced:
        st.warning(
            f"**Resistor Bridge Unbalanced ($R_1 \\ne R_2$ or $R_F \\ne R_3$):** "
            f"Mismatch creates a non-zero common-mode gain $|A_{{CM}}| = {acm_total:.4f}$, reducing CMRR to **{cmrr_db:.2f} dB**. "
            f"Set $R_1 = R_2$ and $R_F = R_3$ to restore full uA741 CMRR ({internal_cmrr_db} dB)."
        )
    else:
        st.success(
            f"**Balanced Differential Amplifier ($R_1 = R_2 = {r1_kohm:.1f}$ kOhm, $R_F = R_3 = {rf_kohm:.1f}$ kOhm):** "
            f"Differential gain **$A_D = -R_F / R_1 = {ad:+.2f}$**, common-mode gain **$|A_{{CM}}| = {acm_total:.2e}$**, "
            f"and **CMRR = {cmrr_db:.2f} dB** at **$f = {freq_hz}$ Hz**."
        )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 3-14*)**")
        fig_sch = draw_diff_amp_schematic(r1_kohm, r2_kohm, rf_kohm, r3_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Gayakwad Eq. (3-25):** $v_o = -\\frac{{R_F}}{{R_1}}(v_x - v_y) = {ad:+.2f}\\,v_{{xy}}$; "
            f"$\\text{{CMRR}} = 20\\log_{{10}}|A_D / A_{{CM}}| = {cmrr_db:.2f}$ dB."
        )

    with cro_col:
        st.markdown(f"**CRO Waveforms at $f = {freq_hz}$ Hz ({op_mode.split(' (')[0]})**")
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        t_ms = t * 1000.0
        w = 2.0 * np.pi * freq_hz

        if op_mode.startswith("Differential"):
            # Differential mode: vx = +0.5*vxy, vy = -0.5*vxy
            vx_t = (vin_test_pp / 4.0) * np.sin(w * t)
            vy_t = -(vin_test_pp / 4.0) * np.sin(w * t)
            vxy_t = vx_t - vy_t
            vo_t = np.clip(ad * vxy_t, -vsat, vsat)

            fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)
            ax_in.plot(t_ms, vx_t, color="#1f77b4", linewidth=1.8, label="Input $v_x(t)$")
            ax_in.plot(t_ms, vy_t, color="#2ca02c", linestyle="--", linewidth=1.8, label="Input $v_y(t)$ (180 deg out of phase)")
            ax_in.plot(t_ms, vxy_t, color="#ff7f0e", linestyle=":", linewidth=1.5, label=f"$v_{{xy}} = v_x - v_y$ ($V_{{pp}} = {vin_test_pp:.2f}$ V)")
            ax_in.set_ylim(-max(vin_test_pp * 0.7, 0.3), max(vin_test_pp * 0.7, 0.3))
            style_cro_axes(ax_in, title="CH1: Differential Inputs $v_x(t), v_y(t)$ & $v_{xy}(t)$", xlabel="", ylabel="Input (V)")
            ax_in.legend(loc="upper right", fontsize=7.5)

            ax_out.plot(t_ms, vo_t, color="#d62728", linewidth=2.2, label=f"Amplified Output $v_o(t)$ ($V_{{o(pp)}} = {np.ptp(vo_t):.2f}$ V)")
            ax_out.set_ylim(-max(np.max(np.abs(vo_t)) * 1.3, 0.5), max(np.max(np.abs(vo_t)) * 1.3, 0.5))
            style_cro_axes(
                ax_out,
                title=f"CH2: Differential Output $v_o(t) = A_D v_{{xy}}(t)$ ($|A_D| = {abs(ad):.2f}$)",
                xlabel="Time (ms)",
                ylabel="Output (V)"
            )
            ax_out.legend(loc="upper right", fontsize=8)
        else:
            # Common mode: vx = vy = vcm
            vcm_t = (vin_test_pp / 2.0) * np.sin(w * t)
            vo_cm_t = np.clip(acm_total * vcm_t, -vsat, vsat)

            fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)
            ax_in.plot(t_ms, vcm_t, color="#1f77b4", linewidth=2.1, label=f"Common Input $v_x(t) = v_y(t) = v_{{cm}}(t)$ ($V_{{pp}} = {vin_test_pp:.2f}$ V)")
            ax_in.set_ylim(-max(vin_test_pp * 0.7, 0.3), max(vin_test_pp * 0.7, 0.3))
            style_cro_axes(ax_in, title="CH1: Common-Mode Input Signal $v_{cm}(t)$", xlabel="", ylabel="Input (V)")
            ax_in.legend(loc="upper right", fontsize=8)

            ax_out.plot(t_ms, vo_cm_t * 1e3, color="#d62728", linewidth=2.2,
                        label=f"Rejected Common-Mode Output $v_o(t)$ ($V_{{o(pp)}} = {np.ptp(vo_cm_t)*1e3:.3f}$ mV)")
            y_cm_lim = max(np.max(np.abs(vo_cm_t * 1e3)) * 1.4, 0.5)
            ax_out.set_ylim(-y_cm_lim, y_cm_lim)
            style_cro_axes(
                ax_out,
                title=f"CH2: Common-Mode Output $v_o(t)$ in mV ($|A_{{CM}}| = {acm_total:.2e}$, CMRR = {cmrr_db:.1f} dB)",
                xlabel="Time (ms)",
                ylabel="Output (mV)"
            )
            ax_out.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Differential & Common Mode Observation Tables + Theory
    # ---------------------------------------------------------------------
    tab_obs, tab_theory = st.tabs([
        "Observation Tables: Differential Mode, Common Mode & CMRR",
        "Theory & Derivation (Gayakwad Sec. 3-5-1)"
    ])

    with tab_obs:
        st.markdown(
            f"Measured output voltages and calculated **CMRR** at **$f = {freq_hz}$ Hz** "
            f"($R_1 = {r1_kohm:.1f}$ kOhm, $R_2 = {r2_kohm:.1f}$ kOhm, $R_F = {rf_kohm:.1f}$ kOhm, $R_3 = {r3_kohm:.1f}$ kOhm):"
        )

        col_d, col_c = st.columns(2, gap="medium")
        with col_d:
            st.markdown("**Table 1: Differential-Mode Gain ($A_D = v_{od} / v_{xy}$)**")
            vxy_list = [0.10, 0.20, 0.50, 0.80, 1.00, 1.20]
            rows_diff = []
            for v_in_d in vxy_list:
                vo_d_id = abs(ad) * v_in_d
                vo_d_ac = min(vo_d_id, 2.0 * vsat)
                ad_meas = vo_d_ac / v_in_d
                rows_diff.append({
                    "Input v_xy(pp) (V)": f"{v_in_d:.2f}",
                    "Output v_od(pp) (V)": f"{vo_d_ac:.3f}",
                    "Gain |AD| = vod/vxy": f"{ad_meas:.2f}",
                    "Status": "Saturated" if vo_d_id > 2.0 * vsat else "Linear"
                })
            st.table(rows_diff)

        with col_c:
            st.markdown("**Table 2: Common-Mode Gain ($A_{CM}$) & CMRR**")
            vcm_list = [0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
            rows_cm = []
            for v_in_c in vcm_list:
                vo_c_ac = min(acm_total * v_in_c, 2.0 * vsat)
                acm_meas = vo_c_ac / v_in_c
                cmrr_m = abs(ad) / max(acm_meas, 1e-12)
                cmrr_m_db = 20.0 * np.log10(cmrr_m)
                rows_cm.append({
                    "Input v_cm(pp) (V)": f"{v_in_c:.2f}",
                    "Output v_ocm(pp) (mV)": f"{vo_c_ac*1e3:.3f}",
                    "Gain |ACM|": f"{acm_meas:.2e}",
                    "CMRR (dB)": f"{cmrr_m_db:.2f} dB"
                })
            st.table(rows_cm)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Derivation (*R. A. Gayakwad, Section 3-5-1, Figure 3-14, p. 97*)

            1. **Superposition Derivation (*Gayakwad Eqs. 3-24a, 3-24b & 3-25*):**
               * Setting $v_y = 0\\text{{ V}}$, the circuit acts as an inverting amplifier with output:
                 $$v_{{ox}} = -\\frac{{R_F}}{{R_1}} v_x \\quad \\text{{(Eq. 3-24a)}}$$
               * Setting $v_x = 0\\text{{ V}}$, the circuit acts as a non-inverting amplifier with voltage divider $v_1 = \\frac{{R_3}}{{R_2 + R_3}} v_y$, giving output:
                 $$v_{{oy}} = \\frac{{R_3}}{{R_2 + R_3}}\\left(\\frac{{R_1 + R_F}}{{R_1}}\\right) v_y$$
               * When **$R_1 = R_2$** and **$R_F = R_3$**, this simplifies to $v_{{oy}} = \\frac{{R_F}}{{R_1}} v_y$ (**Eq. 3-24b**), so the net output voltage is:
                 $$v_o = v_{{ox}} + v_{{oy}} = -\\frac{{R_F}}{{R_1}}(v_x - v_y) = -\\frac{{R_F}}{{R_1}} v_{{xy}}$$
               * Thus the differential voltage gain is (**Eq. 3-25**):
                 $$A_D = \\frac{{v_o}}{{v_{{xy}}}} = -\\frac{{R_F}}{{R_1}} = {ad:+.2f}$$

            2. **Common-Mode Rejection Ratio (CMRR):**
               * In **Common Mode** ($v_x = v_y = v_{{cm}}$), an ideal differential amplifier produces $v_o = 0\\text{{ V}}$. In practice, small resistor mismatches and finite op-amp CMRR produce a small common-mode gain $A_{{CM}} = v_{{ocm}} / v_{{cm}}$.
               * **CMRR** measures the ability to reject common-mode noise while amplifying differential signals:
                 $$\\text{{CMRR}} = \\left|\\frac{{A_D}}{{A_{{CM}}}}\\right| = {cmrr_ratio:,.0f}, \\qquad \\text{{CMRR (dB)}} = 20\\log_{{10}}\\left|\\frac{{A_D}}{{A_{{CM}}}}\\right| = {cmrr_db:.2f}\\text{{ dB}}$$
            """
        )