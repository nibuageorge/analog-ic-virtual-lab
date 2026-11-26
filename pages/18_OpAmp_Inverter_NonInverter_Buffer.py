import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 18: Inverter, Non-Inverter & Voltage Follower", layout="wide")
render_sidebar_header()

st.title("Experiment 18: Op-Amp Inverter, Non-Inverter & Voltage Follower ($\\mu$A741)")
st.markdown(
    "Design and construct **Inverting Amplifier**, **Non-Inverting Amplifier**, and **Voltage Follower (Buffer)** "
    "circuits using **$\\mu$A741** for **AC input voltages**, verify voltage gain ($A_F$) and phase relationship "
    "on the CRO, and compare all three configurations (*Gayakwad, Sections 3-2, 3-3 & 3-3-3, Figs. 3-2, 3-7 & 3-9*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Schematics for All 3 Configurations
# -------------------------------------------------------------------------
def draw_amp_mode_schematic(mode_key, r1_kohm, rf_kohm, rom_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.0)

        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741', loc='center', fontsize=9))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Output node Vo and load RL (common to all 3 circuits)
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center
        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o$', loc='right', fontsize=9.5))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        inv_x, inv_y = op.in1
        noninv_x, noninv_y = op.in2

        if mode_key == "inverter":
            # Gayakwad Fig. 3-2: Inverting Amplifier
            d.add(elm.Line().left().at((inv_x, inv_y)).length(0.75))
            sum_node = d.add(elm.Dot())
            sx, sy = sum_node.center

            # Feedback RF above Op-Amp
            top_y = sy + 1.25
            d.add(elm.Line().up().at((sx, sy)).toy(top_y))
            d.add(elm.Resistor().right().tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
            d.add(elm.Line().down().toy(fb_y))

            # Input R1 and AC source vin at inverting input
            d.add(elm.Resistor().left().at((sx, sy)).length(1.95).label(f'$R_1$\n{r1_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
            src_x = d.here[0]
            d.add(elm.Line().down().at((src_x, sy)).length(0.25))
            d.add(elm.SourceSin().down().length(1.15).label('$v_{in}$', loc='left', fontsize=9))
            d.add(elm.Ground())

            # Non-inverting input grounded through ROM
            d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.45))
            d.add(elm.Resistor().down().length(1.25).label(f'$R_{{OM}}$\n{rom_kohm:.2f} k$\\Omega$', loc='right', fontsize=7.8))
            d.add(elm.Ground())

        elif mode_key == "noninverter":
            # Gayakwad Fig. 3-7: Non-Inverting Amplifier (Zero overlap between vin and R1)
            # 1. AC source vin connects directly to non-inverting (+) input on inner column
            d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.85))
            vin_x = d.here[0]
            d.add(elm.Line().down().at((vin_x, noninv_y)).length(0.2))
            d.add(elm.SourceSin().down().length(1.05).label('$v_{in}$', loc='right', fontsize=9))
            d.add(elm.Ground())

            # 2. Inverting (-) feedback node and R1 routed cleanly on outer column to the left of vin
            d.add(elm.Line().left().at((inv_x, inv_y)).length(0.65))
            fb_node = d.add(elm.Dot())
            sx, sy = fb_node.center

            # Feedback RF above Op-Amp
            top_y = sy + 1.25
            d.add(elm.Line().up().at((sx, sy)).toy(top_y))
            d.add(elm.Resistor().right().tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
            d.add(elm.Line().down().toy(fb_y))

            # Extend horizontal wire left past vin_x before dropping R1 to ground
            r1_x = vin_x - 1.15
            d.add(elm.Line().left().at((sx, sy)).tox(r1_x))
            d.add(elm.Resistor().down().at((r1_x, sy)).length(1.85).label(f'$R_1$\n{r1_kohm:.1f} k$\\Omega$', loc='left', fontsize=8.2))
            d.add(elm.Ground())

        else:
            # Gayakwad Fig. 3-9: Voltage Follower (Unity-Gain Buffer, RF = 0, R1 = inf)
            d.add(elm.Line().left().at((inv_x, inv_y)).length(0.75))
            sx, sy = d.here
            top_y = sy + 1.25
            d.add(elm.Line().up().at((sx, sy)).toy(top_y))
            d.add(elm.Line().right().tox(fb_x).label('Direct Feedback ($R_F = 0\\,\\Omega$)', loc='top', fontsize=7.8))
            d.add(elm.Line().down().toy(fb_y))

            # AC source vin directly at non-inverting (+) input
            d.add(elm.Line().left().at((noninv_x, noninv_y)).length(1.45))
            src_x = d.here[0]
            d.add(elm.Line().down().at((src_x, noninv_y)).length(0.2))
            d.add(elm.SourceSin().down().length(1.1).label('$v_{in}$', loc='left', fontsize=9))
            d.add(elm.Ground())

        return d.draw(show=False).fig
# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Select Amplifier Configuration")
    config_choice = st.radio(
        "Circuit Mode",
        [
            "Inverting Amplifier (AF = -RF / R1)",
            "Non-Inverting Amplifier (AF = 1 + RF / R1)",
            "Voltage Follower / Buffer (AF = +1)",
        ],
        index=0,
    )

    if config_choice.startswith("Inverting"):
        mode_key = "inverter"
    elif config_choice.startswith("Non-Inverting"):
        mode_key = "noninverter"
    else:
        mode_key = "buffer"

    st.subheader("2. Feedback & Input Resistors")
    is_buffer = (mode_key == "buffer")
    r1_kohm = st.slider(
        "Input / Divider Resistor $R_1$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        disabled=is_buffer,
        help="Not used in Voltage Follower (R1 = open)."
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kOhm)",
        min_value=1.0,
        max_value=200.0,
        value=47.0,
        step=1.0,
        disabled=is_buffer,
        help="Not used in Voltage Follower (RF = 0 Ohm direct short)."
    )
    rl_kohm = st.slider("Load Resistor $R_L$ (kOhm)", 1.0, 50.0, 10.0, 1.0)

    st.subheader("3. AC Input Signal & Supply")
    vin_pp = st.slider("AC Input Voltage $V_{in(pp)}$ (V)", 0.2, 10.0, 2.0, 0.2)
    freq_hz = st.slider("AC Signal Frequency $f$ (Hz)", 100, 5000, 1000, 100)
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Gain, Phase & Output Calculations (Gayakwad Chapter 3)
# -------------------------------------------------------------------------
vsat = max(vcc - 1.0, 0.5)
rom_kohm = (r1_kohm * rf_kohm) / (r1_kohm + rf_kohm)

if mode_key == "inverter":
    af_ideal = -(rf_kohm / r1_kohm)
    phase_deg = 180
    formula_str = f"A_F = -R_F / R_1 = -{rf_kohm:.0f} / {r1_kohm:.0f} = {af_ideal:.2f}"
elif mode_key == "noninverter":
    af_ideal = 1.0 + (rf_kohm / r1_kohm)
    phase_deg = 0
    formula_str = f"A_F = 1 + R_F / R_1 = 1 + {rf_kohm:.0f} / {r1_kohm:.0f} = +{af_ideal:.2f}"
else:
    af_ideal = 1.0
    phase_deg = 0
    formula_str = "A_F = +1.00 (Unity-Gain Buffer, R_F = 0, R_1 = inf)"

vo_ideal_pp = abs(af_ideal) * vin_pp
vo_actual_pp = min(vo_ideal_pp, 2.0 * vsat)
af_measured = (vo_actual_pp / vin_pp) * (-1.0 if af_ideal < 0 else 1.0)
saturated = vo_ideal_pp > (2.0 * vsat)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, CRO Waveforms & Comparison Tabs
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed-Loop Gain ($A_F$)", f"{af_ideal:+.2f}")
    m2.metric("Input $V_{in(pp)}$", f"{vin_pp:.2f} V")
    m3.metric("Output $V_{o(pp)}$", f"{vo_actual_pp:.2f} V")
    m4.metric("Phase Shift ($\\Delta\\phi$)", f"{phase_deg} deg")

    if saturated:
        st.warning(
            f"**Output Saturation Clipping ($V_{{o,\\text{{ideal}}(pp)}} = {vo_ideal_pp:.2f}$ V > $2V_{{sat}} = {2*vsat:.1f}$ V):** "
            f"Output peaks are clipped at $\\pm {vsat:.1f}$ V ($V_{{o(pp)}} = {vo_actual_pp:.2f}$ V). Reduce $V_{{in(pp)}}$ or $R_F$ for undistorted sine wave."
        )
    else:
        st.success(
            f"**Linear AC Amplification (${formula_str}$):** "
            f"For $V_{{in(pp)}} = {vin_pp:.2f}$ V at $f = {freq_hz}$ Hz, output $V_{{o(pp)}} = {vo_actual_pp:.2f}$ V "
            f"with **{phase_deg} deg** phase shift."
        )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        fig_label = {"inverter": "Fig. 3-2", "noninverter": "Fig. 3-7", "buffer": "Fig. 3-9"}[mode_key]
        st.markdown(f"**Circuit Diagram (*Gayakwad {fig_label}*)**")
        fig_sch = draw_amp_mode_schematic(mode_key, r1_kohm, rf_kohm, rom_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(f"**Design Equation:** ${formula_str}$; Phase Shift $= {phase_deg}^\\circ$.")

    with cro_col:
        st.markdown(f"**CRO Waveforms at $f = {freq_hz}$ Hz ({config_choice.split(' (')[0]})**")
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        t_ms = t * 1000.0
        w = 2.0 * np.pi * freq_hz

        vin_t = (vin_pp / 2.0) * np.sin(w * t)
        vo_t = np.clip(af_ideal * vin_t, -vsat, vsat)

        fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)

        ax_in.plot(t_ms, vin_t, color="#1f77b4", linewidth=2.1, label=f"CH1: AC Input $v_{{in}}(t)$ ($V_{{pp}} = {vin_pp:.2f}$ V)")
        in_lim = max(vin_pp * 0.75, 0.5)
        ax_in.set_ylim(-in_lim, in_lim)
        style_cro_axes(ax_in, title=f"CH1: Sinusoidal Input Signal $v_{{in}}(t)$ ($f = {freq_hz}$ Hz)", xlabel="", ylabel="$v_{in}$ (V)")
        ax_in.legend(loc="upper right", fontsize=8)

        ax_out.plot(
            t_ms,
            vo_t,
            color="#d62728",
            linewidth=2.2,
            label=f"CH2: Output $v_o(t)$ ($V_{{pp}} = {vo_actual_pp:.2f}$ V, $\\Delta\\phi = {phase_deg}^\\circ$)"
        )
        out_lim = max(vo_actual_pp * 0.75, 0.5)
        ax_out.set_ylim(-out_lim, out_lim)
        style_cro_axes(
            ax_out,
            title=f"CH2: Output Waveform $v_o(t) = A_F v_{{in}}(t)$ ($A_F = {af_ideal:+.2f}$)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Side-by-Side Overlay + Observation Tables + Theory
    # ---------------------------------------------------------------------
    tab_cmp, tab_obs, tab_theory = st.tabs([
        "3-Mode Simultaneous Overlay Plot",
        "Observation Tables (Inverter, Non-Inverter & Buffer)",
        "Theory & Summary Table (Gayakwad Ch. 3)"
    ])

    with tab_cmp:
        st.markdown(
            f"Simultaneous output comparison for **$V_{{in(pp)}} = {vin_pp:.2f}$ V**, **$R_1 = {r1_kohm:.0f}$ kOhm**, "
            f"**$R_F = {rf_kohm:.0f}$ kOhm** at **$f = {freq_hz}$ Hz**:"
        )
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 2.5 * T, 1600)
        t_ms = t * 1000.0
        vin_t = (vin_pp / 2.0) * np.sin(2.0 * np.pi * freq_hz * t)

        vo_inv = np.clip(-(rf_kohm / r1_kohm) * vin_t, -vsat, vsat)
        vo_non = np.clip((1.0 + rf_kohm / r1_kohm) * vin_t, -vsat, vsat)
        vo_buf = np.clip(1.0 * vin_t, -vsat, vsat)

        fig_all, ax_all = plt.subplots(figsize=(8.6, 3.8))
        ax_all.plot(t_ms, vin_t, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Input $v_{{in}}(t)$ ({vin_pp:.2f} V pp)")
        ax_all.plot(t_ms, vo_buf, color="#2ca02c", linewidth=2.0, label="Voltage Follower ($A_F = +1.00$, $0^\\circ$)")
        ax_all.plot(t_ms, vo_inv, color="#d62728", linewidth=2.1, label=f"Inverting Amp ($A_F = -{rf_kohm/r1_kohm:.2f}$, $180^\\circ$)")
        ax_all.plot(t_ms, vo_non, color="#9467bd", linewidth=2.1, label=f"Non-Inverting Amp ($A_F = +{1+rf_kohm/r1_kohm:.2f}$, $0^\\circ$)")
        style_cro_axes(
            ax_all,
            title="Comparison of Output Waveforms: Inverter vs. Non-Inverter vs. Voltage Follower",
            xlabel="Time (ms)",
            ylabel="Voltage (V)"
        )
        ax_all.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_all)
        plt.close(fig_all)

    with tab_obs:
        st.markdown(
            f"Measured output peak-to-peak voltages across AC input amplitudes for **$R_1 = {r1_kohm:.0f}$ kOhm**, "
            f"**$R_F = {rf_kohm:.0f}$ kOhm** ($\\pm V_{{sat}} = \\pm {vsat:.1f}$ V):"
        )
        vin_steps = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        obs_rows = []
        for v_in_s in vin_steps:
            vo_i = min((rf_kohm / r1_kohm) * v_in_s, 2.0 * vsat)
            vo_n = min((1.0 + rf_kohm / r1_kohm) * v_in_s, 2.0 * vsat)
            vo_b = min(1.0 * v_in_s, 2.0 * vsat)
            obs_rows.append({
                "AC Input Vin(pp) (V)": f"{v_in_s:.2f}",
                "Inverter Vo(pp) [180 deg] (V)": f"{vo_i:.2f}",
                "Non-Inverter Vo(pp) [0 deg] (V)": f"{vo_n:.2f}",
                "Buffer Vo(pp) [0 deg] (V)": f"{vo_b:.2f}",
            })
        st.table(obs_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Comparison Summary (*R. A. Gayakwad, Chapter 3*)

            | Parameter | Inverting Amplifier (*Fig. 3-2*) | Non-Inverting Amplifier (*Fig. 3-7*) | Voltage Follower / Buffer (*Fig. 3-9*) |
            | :--- | :--- | :--- | :--- |
            | **Input Terminal** | Inverting (`-`) via $R_1$ | Non-Inverting (`+`) direct | Non-Inverting (`+`) direct |
            | **Closed-Loop Gain ($A_F$)** | $-\\dfrac{{R_F}}{{R_1}} = -{rf_kohm/r1_kohm:.2f}$ | $1 + \\dfrac{{R_F}}{{R_1}} = +{1 + rf_kohm/r1_kohm:.2f}$ | $+1.00$ ($R_F = 0$, $R_1 = \\infty$) |
            | **Phase Shift ($v_o$ vs. $v_{{in}}$)** | **$180^\\circ$** (Out of phase) | **$0^\\circ$** (In phase) | **$0^\\circ$** (In phase) |
            | **Input Resistance ($R_{{iF}}$)** | $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$ | $R_i(1 + A_{{OL}}B)$ (Very high, $>100\\text{{ M}}\\Omega$) | $A_{{OL}} R_i$ (Highest possible) |
            | **Primary Application** | Phase inversion, scaling, summing | High-$Z_{{in}}$ voltage amplification | Impedance matching & isolation |
            """
        )
