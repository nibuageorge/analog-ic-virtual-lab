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
    "Design and construct a **Differential Amplifier** using **$\\mu$A741** "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 3-14 Differential Amplifier Schematic
# -------------------------------------------------------------------------
def draw_diff_amp_schematic(r1_kohm, rf_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741', loc='center', fontsize=9))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Output Node vo and Load Resistor RL
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center
        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o$', loc='right', fontsize=9.5))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Upper Inverting Branch: vx -> R1 -> node v2 -> RF -> Output
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.15
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.55))
        d.add(elm.Line().up().toy(top_y))
        v2_node = d.add(elm.Dot().label('$v_2$', loc='top', fontsize=8.5))
        v2_x, v2_y = v2_node.center

        d.add(elm.Resistor().right().at((v2_x, v2_y)).tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
        d.add(elm.Line().down().toy(fb_y))

        d.add(elm.Resistor().left().at((v2_x, v2_y)).length(1.95).label(f'$R_1$\n{r1_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, v2_y)).length(0.2))
        d.add(elm.SourceSin().down().length(1.05).label('$v_x$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # Lower Non-Inverting Branch: vy -> R2 (=R1) -> node v1 -> R3 (=RF) -> Ground
        noninv_x, noninv_y = op.in2
        bot_y = noninv_y - 0.55
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.55))
        d.add(elm.Line().down().toy(bot_y))
        v1_node = d.add(elm.Dot().label('$v_1$', loc='top', fontsize=8.5))
        v1_x, v1_y = v1_node.center

        d.add(elm.Resistor().down().at((v1_x, v1_y)).length(1.35).label(f'$R_3 = R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='right', fontsize=8))
        d.add(elm.Ground())

        d.add(elm.Resistor().left().at((v1_x, v1_y)).length(1.95).label(f'$R_2 = R_1$\n{r1_kohm:.1f} k$\\Omega$', loc='top', fontsize=8))
        src_y_x = d.here[0]
        d.add(elm.Line().down().at((src_y_x, v1_y)).length(0.2))
        d.add(elm.SourceSin().down().length(1.05).label('$v_y$', loc='left', fontsize=9))
        d.add(elm.Ground())

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Waveforms)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Circuit Resistors ($R_1 = R_2, R_F = R_3$)")
    r1_kohm = st.slider("Input Resistors $R_1 = R_2$ (kOhm)", 1.0, 50.0, 10.0, 1.0)
    rf_kohm = st.slider("Feedback & Ground Resistors $R_F = R_3$ (kOhm)", 10.0, 200.0, 100.0, 5.0)
    rl_kohm = st.slider("Load Resistor $R_L$ (kOhm)", 1.0, 50.0, 10.0, 1.0)

    st.subheader("2. Test Mode & Input Voltages")
    op_mode = st.radio(
        "Select Measurement Mode",
        ["Differential Mode (Measure AD = vod / vd)", "Common Mode (Measure ACM = vocm / vcm)"],
        index=0
    )
    vd_pp = st.slider(
        "Differential Input Voltage $v_d$ (V pp)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Input difference vd = vx - vy applied in Differential Mode."
    )
    vcm_pp = st.slider(
        "Common-Mode Input Voltage $v_{cm}$ (V pp)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Common input vx = vy = vcm applied in Common Mode."
    )
    freq_hz = st.slider("Signal Frequency $f$ (Hz)", 100, 5000, 1000, 100)

    st.subheader("3. Practical Common-Mode Gain & Supply")
    acm_practical = st.selectbox(
        "Practical Common-Mode Gain $A_{CM}$ of Setup",
        options=[0.0005, 0.0010, 0.0020, 0.0050],
        index=1,
        format_func=lambda x: f"{x:.4f} ({x*1e3:.1f} mV output per 1 V input)",
        help="In a real lab circuit, slight resistor tolerances give a small ACM (typically 0.0005 to 0.005)."
    )
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Simple 3-Step Lab Calculations
# -------------------------------------------------------------------------
vsat = max(vcc - 1.0, 0.5)

# Step 1: Differential Mode
ad_mag = rf_kohm / r1_kohm
vod_pp = min(ad_mag * vd_pp, 2.0 * vsat)
ad_measured = vod_pp / vd_pp

# Step 2: Common Mode
vocm_pp = min(acm_practical * vcm_pp, 2.0 * vsat)
acm_measured = vocm_pp / vcm_pp

# Step 3: CMRR
cmrr_ratio = ad_measured / acm_measured
cmrr_db = 20.0 * np.log10(cmrr_ratio)

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Differential Gain ($A_D = v_{od}/v_d$)", f"{ad_measured:.2f}")
    m2.metric("Common-Mode Gain ($A_{CM} = v_{ocm}/v_{cm}$)", f"{acm_measured:.4f}")
    m3.metric("CMRR ($A_D / A_{CM}$)", f"{cmrr_ratio:,.0f}")
    m4.metric("CMRR in dB ($20\\log_{10}$)", f"{cmrr_db:.2f} dB")

    st.info(
        f"**Step-by-Step Calculation at $f = {freq_hz}$ Hz:**  \n"
        f"1. **Differential Gain:** $A_D = \\frac{{v_{{od}}}}{{v_d}} = \\frac{{{vod_pp:.2f}\\text{{ V}}}}{{{vd_pp:.2f}\\text{{ V}}}} = {ad_measured:.2f}$  \n"
        f"2. **Common-Mode Gain:** $A_{{CM}} = \\frac{{v_{{ocm}}}}{{v_{{cm}}}} = \\frac{{{vocm_pp*1e3:.2f}\\text{{ mV}}}}{{{vcm_pp:.2f}\\text{{ V}}}} = {acm_measured:.4f}$  \n"
        f"3. **CMRR:** $\\text{{CMRR}} = \\frac{{A_D}}{{A_{{CM}}}} = \\frac{{{ad_measured:.2f}}}{{{acm_measured:.4f}}} = {cmrr_ratio:,.0f} \\implies 20\\log_{{10}}({cmrr_ratio:,.0f}) = {cmrr_db:.2f}\\text{{ dB}}$"
    )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_diff_amp_schematic(r1_kohm, rf_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(f"$R_1 = R_2 = {r1_kohm:.0f}$ kOhm, $R_F = R_3 = {rf_kohm:.0f}$ kOhm -> $A_D = R_F / R_1 = {ad_mag:.1f}$.")

    with cro_col:
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        t_ms = t * 1000.0
        w = 2.0 * np.pi * freq_hz

        if op_mode.startswith("Differential"):
            st.markdown(f"**Differential Mode CRO Waveforms ($f = {freq_hz}$ Hz)**")
            vd_t = (vd_pp / 2.0) * np.sin(w * t)
            vod_t = np.clip(-ad_mag * vd_t, -vsat, vsat)

            fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
            ax_in.plot(t_ms, vd_t, color="#1f77b4", linewidth=2.1, label=f"Differential Input $v_d$ ({vd_pp:.2f} V pp)")
            ax_in.set_ylim(-max(vd_pp * 0.75, 0.3), max(vd_pp * 0.75, 0.3))
            style_cro_axes(ax_in, title=f"CH1: Differential Input $v_d = v_x - v_y$ ($v_d = {vd_pp:.2f}$ V pp)", xlabel="", ylabel="$v_d$ (V)")
            ax_in.legend(loc="upper right", fontsize=8)

            ax_out.plot(t_ms, vod_t, color="#d62728", linewidth=2.2, label=f"Differential Output $v_{{od}}$ ({vod_pp:.2f} V pp)")
            ax_out.set_ylim(-max(vod_pp * 0.75, 0.5), max(vod_pp * 0.75, 0.5))
            style_cro_axes(ax_out, title=f"CH2: Amplified Output $v_{{od}}$ ($A_D = v_{{od}}/v_d = {ad_measured:.2f}$)", xlabel="Time (ms)", ylabel="$v_{od}$ (V)")
            ax_out.legend(loc="upper right", fontsize=8)
        else:
            st.markdown(f"**Common Mode CRO Waveforms ($f = {freq_hz}$ Hz)**")
            vcm_t = (vcm_pp / 2.0) * np.sin(w * t)
            vocm_t = np.clip(acm_practical * vcm_t, -vsat, vsat)

            fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
            ax_in.plot(t_ms, vcm_t, color="#1f77b4", linewidth=2.1, label=f"Common Input $v_{{cm}}$ ({vcm_pp:.2f} V pp)")
            ax_in.set_ylim(-max(vcm_pp * 0.75, 0.5), max(vcm_pp * 0.75, 0.5))
            style_cro_axes(ax_in, title=f"CH1: Common-Mode Input $v_x = v_y = v_{{cm}}$ ($v_{{cm}} = {vcm_pp:.2f}$ V pp)", xlabel="", ylabel="$v_{cm}$ (V)")
            ax_in.legend(loc="upper right", fontsize=8)

            ax_out.plot(t_ms, vocm_t * 1e3, color="#d62728", linewidth=2.2, label=f"Common Output $v_{{ocm}}$ ({vocm_pp*1e3:.2f} mV pp)")
            ax_out.set_ylim(-max(vocm_pp * 1e3 * 0.85, 0.5), max(vocm_pp * 1e3 * 0.85, 0.5))
            style_cro_axes(ax_out, title=f"CH2: Common-Mode Output $v_{{ocm}}$ ($A_{{CM}} = v_{{ocm}}/v_{{cm}} = {acm_measured:.4f}$)", xlabel="Time (ms)", ylabel="$v_{ocm}$ (mV)")
            ax_out.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Observation Tables Tab
    # ---------------------------------------------------------------------
    tab_obs, tab_theory = st.tabs([
        "Observation Tables (Differential Mode, Common Mode & CMRR)",
        "Theory & Formula Summary"
    ])

    with tab_obs:
        col_d, col_c = st.columns(2, gap="medium")
        with col_d:
            st.markdown("**Table 1: Differential Mode ($A_D = v_{od} / v_d$)**")
            rows_d = []
            for v_in_d in [0.2, 0.4, 0.6, 0.8, 1.0]:
                v_out_d = min(ad_mag * v_in_d, 2.0 * vsat)
                rows_d.append({
                    "Input vd (V pp)": f"{v_in_d:.2f}",
                    "Output vod (V pp)": f"{v_out_d:.2f}",
                    "AD = vod / vd": f"{v_out_d / v_in_d:.2f}"
                })
            st.table(rows_d)

        with col_c:
            st.markdown("**Table 2: Common Mode ($A_{CM} = v_{ocm} / v_{cm}$) & CMRR**")
            rows_c = []
            for v_in_c in [1.0, 2.0, 3.0, 4.0, 5.0]:
                v_out_c = acm_practical * v_in_c
                acm_val = v_out_c / v_in_c
                cmrr_val = ad_measured / acm_val
                rows_c.append({
                    "Input vcm (V pp)": f"{v_in_c:.2f}",
                    "Output vocm (mV pp)": f"{v_out_c*1e3:.2f}",
                    "ACM = vocm / vcm": f"{acm_val:.4f}",
                    "CMRR (dB)": f"{20*np.log10(cmrr_val):.2f} dB"
                })
            st.table(rows_c)

    with tab_theory:
        st.markdown(
            f"""
            ### Working Formulas (R. A. Gayakwad)

            1. **Output Voltage of Differential Amplifier:**
               When $R_1 = R_2$ and $R_F = R_3$:
               $$v_o = -\\frac{{R_F}}{{R_1}}(v_x - v_y)$$

            2. **Differential-Mode Gain ($A_D$):**
               $$A_D = \\frac{{v_{{od}}}}{{v_d}} = \\frac{{R_F}}{{R_1}}$$

            3. **Common-Mode Gain ($A_{{CM}}$):**
               With both inputs shorted together ($v_x = v_y = v_{{cm}}$):
               $$A_{{CM}} = \\frac{{v_{{ocm}}}}{{v_{{cm}}}}$$

            4. **Common-Mode Rejection Ratio ($\text{{CMRR}}$):**
               $$\\text{{CMRR}} = \\frac{{A_D}}{{A_{{CM}}}}, \\qquad \\text{{CMRR (dB)}} = 20 \\log_{{10}}\\left(\\frac{{A_D}}{{A_{{CM}}}}\\right)$$
            """
        )
