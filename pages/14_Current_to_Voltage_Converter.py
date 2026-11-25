import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 14: Current-to-Voltage Converter", layout="wide")
render_sidebar_header()

st.title("Experiment 14: Current-to-Voltage Converter using Op-Amp ($\\mu$A741)")
st.markdown(
    "Design and construct a **Current-to-Voltage (I-to-V) Converter** using **$\\mu$A741**, "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 3-12 Current-to-Voltage Schematic
# -------------------------------------------------------------------------
def draw_i_to_v_schematic(rf_kohm, rl_kohm, iin_ma, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.3)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Non-Inverting (+) Terminal Tied Directly to Ground (v1 = 0 V, Fig. 3-12) ---
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.55).label('$v_1 = 0$ V', loc='top', fontsize=7.8))
        d.add(elm.Line().down().length(0.75).label('$I_{B1} \\cong 0$', loc='bottom', fontsize=7.5))
        d.add(elm.Ground())

        # --- Output Node vo = -iin*RF and Load Resistor RL ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center
        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o = -i_{in} R_F$', loc='right', fontsize=8.8))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # --- Inverting (-) Input Node v2 (Virtual Ground) & Current Source i_in ---
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.75).label('$I_{B2} \\cong 0$', loc='bottom', fontsize=7.5))
        sum_node = d.add(elm.Dot().label('$v_2$', loc='top', fontsize=8.5))
        sx, sy = sum_node.center

        # Input Current Source i_in connected to node v2
        d.add(elm.Line().left().at((sx, sy)).length(1.45).label(f'$i_{{in}}$ = {iin_ma:+.2f} mA', loc='top', fontsize=8))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, sy)).length(0.25))
        d.add(elm.SourceI().down().reverse().length(1.15).label('$i_{in}$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # --- Feedback Resistor RF ---
        rf_y = sy + 1.15
        d.add(elm.Line().up().at((sx, sy)).toy(rf_y))
        d.add(elm.Resistor().right().tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.5))
        d.add(elm.Line().down().toy(fb_y))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Feedback & Load Resistors")
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Sets conversion sensitivity vo / iin = -RF (default 10 kOhm -> -10 V/mA)."
    )
    rl_kohm = st.slider(
        "Load Resistor $R_L$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    st.subheader("2. Input Current Signal ($i_{in}$)")
    iin_dc_ma = st.slider(
        "DC Input Current $i_{in}$ (mA)",
        min_value=-2.00,
        max_value=2.00,
        value=0.50,
        step=0.05,
        help="Input current fed into inverting terminal node v2 (virtual ground)."
    )
    iin_ac_peak_ma = st.slider(
        "AC Peak Current $I_p$ for CRO (mA)",
        min_value=0.10,
        max_value=2.00,
        value=0.50,
        step=0.05
    )
    freq_hz = st.slider(
        "AC Test Frequency $f$ (Hz)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

    st.subheader("3. Power Supply")
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 3-4-7, Eq. 3-23)
# -------------------------------------------------------------------------
Rf = rf_kohm * 1e3
vsat = max(vcc - 1.0, 0.5)

iin_amp = iin_dc_ma * 1e-3
vo_ideal = -iin_amp * Rf
vo_actual = float(np.clip(vo_ideal, -vsat, vsat))
saturated = abs(vo_ideal) > vsat
iin_max_ma = (vsat / Rf) * 1e3

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, Transfer Curve & CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input Current ($i_{in}$)", f"{iin_dc_ma:+.2f} mA")
    m2.metric("Output Voltage ($v_o$)", f"{vo_actual:+.2f} V")
    m3.metric("Sensitivity ($v_o / i_{in}$)", f"-{rf_kohm:.1f} V/mA")
    m4.metric("Linear Range ($i_{in(max)}$)", f"+/- {iin_max_ma:.2f} mA")

    if saturated:
        st.warning(
            f"**Op-Amp Output Saturated ($|v_{{o(ideal)}}| = {abs(vo_ideal):.2f}$ V > $V_{{sat}} = {vsat:.1f}$ V):** "
            f"For $R_F = {rf_kohm:.1f}$ kOhm, linear conversion holds up to **$\\pm {iin_max_ma:.2f}$ mA**. "
            f"Reduce $|i_{{in}}|$ or lower $R_F$."
        )
    else:
        st.success(
            f"**Linear Current-to-Voltage Conversion ($v_o = -i_{{in}} R_F$):** "
            f"Input **$i_{{in}} = {iin_dc_ma:+.2f}$ mA** across **$R_F = {rf_kohm:.1f}$ kOhm** produces "
            f"**$v_o = {vo_actual:+.2f}$ V** with virtual ground $v_1 = v_2 = 0$ V."
        )

    schem_col, trans_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_i_to_v_schematic(rf_kohm, rl_kohm, iin_dc_ma, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Gayakwad Eq. (3-23):** $v_o = -i_{{in}} R_F$; $v_1 = 0$ V, $v_1 = v_2$; "
            f"$i_{{in(\\max)}} = V_{{sat}}/R_F = {iin_max_ma:.2f}\\text{{ mA}}$."
        )

    with trans_col:
        st.markdown("**DC Transfer Characteristic ($v_o$ vs. $i_{in}$)**")
        i_sweep_ma = np.linspace(-2.2, 2.2, 500)
        vo_sweep_ideal = -(i_sweep_ma * 1e-3) * Rf
        vo_sweep_clip = np.clip(vo_sweep_ideal, -vsat, vsat)

        fig_tr, ax_tr = plt.subplots(figsize=(7.6, 4.9))
        ax_tr.plot(i_sweep_ma, vo_sweep_clip, color="#1f77b4", linewidth=2.4, label=f"Output $v_o$ ($R_F = {rf_kohm:.1f}$ kOhm)")
        ax_tr.plot(i_sweep_ma, vo_sweep_ideal, color="#888888", linestyle=":", linewidth=1.1, label="Ideal Unclipped $-i_{in} R_F$")
        ax_tr.axhline(vsat, color="#d62728", linestyle="--", linewidth=1.0, label=f"$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V")
        ax_tr.axhline(-vsat, color="#d62728", linestyle="--", linewidth=1.0)
        ax_tr.scatter([iin_dc_ma], [vo_actual], color="#d62728", zorder=6, s=55,
                     label=f"Operating Pt ({iin_dc_ma:+.2f} mA, {vo_actual:+.2f} V)")
        ax_tr.set_xlim(-2.2, 2.2)
        ax_tr.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_tr,
            title=f"Linear I-to-V Transfer Curve (Slope = -{rf_kohm:.1f} V/mA)",
            xlabel="Input Current $i_{in}$ (mA)",
            ylabel="Output Voltage $v_o$ (V)"
        )
        ax_tr.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_tr)
        plt.close(fig_tr)

    tab_cro, tab_obs, tab_theory = st.tabs([
        "AC Current-to-Voltage Waveforms (CRO)",
        "DC Observation Table (vo vs. iin)",
        "Theory & Derivation (Gayakwad Sec. 3-4-7)"
    ])

    with tab_cro:
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        w = 2.0 * np.pi * freq_hz
        iin_t_ma = iin_ac_peak_ma * np.sin(w * t)
        vo_t = np.clip(-(iin_ac_peak_ma * 1e-3) * Rf * np.sin(w * t), -vsat, vsat)

        fig_cro, (ax_i, ax_v) = plt.subplots(2, 1, figsize=(8.8, 4.6), sharex=True)
        t_ms = t * 1000.0

        ax_i.plot(t_ms, iin_t_ma, color="#1f77b4", linewidth=2.1, label=f"Input Current $i_{{in}}(t)$ ($I_p = {iin_ac_peak_ma:.2f}$ mA)")
        ax_i.set_ylim(-max(iin_ac_peak_ma * 1.35, 0.5), max(iin_ac_peak_ma * 1.35, 0.5))
        style_cro_axes(
            ax_i,
            title=f"CH1: AC Input Current $i_{{in}}(t)$ ($f = {freq_hz}$ Hz)",
            xlabel="",
            ylabel="$i_{in}$ (mA)"
        )
        ax_i.legend(loc="upper right", fontsize=8.5)

        ax_v.plot(t_ms, vo_t, color="#d62728", linewidth=2.2, label="Output Voltage $v_o(t) = -i_{in}(t) R_F$ (180 deg inverted)")
        ax_v.axhline(vsat, color="#888888", linestyle=":", linewidth=1.0)
        ax_v.axhline(-vsat, color="#888888", linestyle=":", linewidth=1.0)
        ax_v.set_ylim(-vsat * 1.2, vsat * 1.2)
        style_cro_axes(
            ax_v,
            title=f"CH2: Converted Output Voltage $v_o(t)$ ($V_{{o(pp)}} = {np.ptp(vo_t):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_v.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    with tab_obs:
        st.markdown(
            f"Tabulated DC output voltage readings for **$R_F = {rf_kohm:.1f}$ kOhm** "
            f"(Saturation limit **$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V**):"
        )
        i_table_vals = [-1.50, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.50]
        obs_rows = []
        for i_val in i_table_vals:
            v_id = -(i_val * 1e-3) * Rf
            v_ac = float(np.clip(v_id, -vsat, vsat))
            obs_rows.append({
                "Input Current i_in (mA)": f"{i_val:+.2f}",
                "Ideal Output -i_in*RF (V)": f"{v_id:+.2f}",
                "Measured Output vo (V)": f"{v_ac:+.2f}",
                "Ratio |vo / i_in| (V/mA)": f"{abs(v_ac / i_val):.2f}" if i_val != 0 else f"{rf_kohm:.2f}",
                "Operating State": "Saturated" if abs(v_id) > vsat else "Linear"
            })
        st.table(obs_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Derivation (R. A. Gayakwad)

            1. **Derivation from Inverting Amplifier (Gayakwad):**
               Starting with the ideal voltage-gain equation of an inverting amplifier:
               $$\\frac{{v_o}}{{v_{{in}}}} = -\\frac{{R_F}}{{R_1}} \\implies v_o = -\\left(\\frac{{v_{{in}}}}{{R_1}}\\right) R_F$$
               Since the non-inverting terminal is grounded ($v_1 = 0\\text{{ V}}$) and $v_1 = v_2$ (virtual ground), the input current is $\\frac{{v_{{in}}}}{{R_1}} = i_{{in}}$. Replacing the $v_{{in}}$ and $R_1$ combination by a current source $i_{{in}}$ (with $I_{{B1}} \\cong 0$, $I_{{B2}} \\cong 0$) gives:
               $$v_o = -i_{{in}} R_F = -({iin_dc_ma:+.2f}\\text{{ mA}})({rf_kohm:.1f}\\text{{ k}}\\Omega) = {vo_ideal:+.2f}\\text{{ V}}$$

            2. **Practical Applications:**
               * Sensing current from **photodetectors** (photodiodes and photocells) where output voltage is directly proportional to light-induced current.
               * Output stage in **digital-to-analog converters (DACs)** to convert weighted ladder currents into a proportional analog voltage.
            """
        )
