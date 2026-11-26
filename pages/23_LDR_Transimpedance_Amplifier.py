import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 23: Photoconductor (LDR) Transimpedance Amplifier", layout="wide")
render_sidebar_header()

st.title("Experiment 23: Photoconductor (LDR) Transimpedance Amplifier")
st.markdown(
    "**Aim:** To design and construct a **Photoconductor (LDR) based Transimpedance Amplifier** using an operational "
    "amplifier (**LM358**) and study its performance by observing the output voltage ($V_O$) variation with change in light intensity."
)

# -------------------------------------------------------------------------
# Helper: Draw LDR Transimpedance Amplifier Schematic
# -------------------------------------------------------------------------
def draw_ldr_tia_schematic(vbias_v, rsen_kohm, rf_kohm, rcom_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.0)

        # Op-Amp (LM358)
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label("LM358", loc="center", fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.32).label(f"+{vcc:.0f}V", loc="top", fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.32).label(f"-{vcc:.0f}V", loc="bottom", fontsize=7.5))

        # Output terminal Vo
        d.add(elm.Line().right().at(op.out).length(0.75))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center
        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85).label("$V_O$", loc="right", fontsize=9.5))

        # Inverting input (-) summing node
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.75))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # Feedback resistor Rf above op-amp
        top_y = sy + 1.25
        d.add(elm.Line().up().at((sx, sy)).toy(top_y))
        d.add(elm.Resistor().right().tox(fb_x).label(f"$R_f$\n{rf_kohm:.1f} k$\\Omega$", loc="top", fontsize=8.2))
        d.add(elm.Line().down().toy(fb_y))

        # LDR (Rsen) and DC bias source Vbias at inverting (-) input
        d.add(
            elm.Photoresistor()
            .left()
            .at((sx, sy))
            .length(2.05)
            .label(f"LDR ($R_{{sen}}$)\n{rsen_kohm:.2f} k$\\Omega$", loc="top", fontsize=8.0)
        )
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, sy)).length(0.2))
        d.add(elm.SourceV().down().length(1.15).label(f"$V_{{bias}}$\n{vbias_v:.1f} V", loc="left", fontsize=8.2))
        d.add(elm.Ground())

        # Non-inverting (+) input grounded through offset compensation resistor Rcom
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.45))
        d.add(
            elm.Resistor()
            .down()
            .length(1.25)
            .label(f"$R_{{com}}$\n{rcom_kohm:.2f} k$\\Omega$", loc="right", fontsize=7.8)
        )
        d.add(elm.Ground())

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Physics Model: Photoconductor (LDR) Resistance & Transimpedance Output
# -------------------------------------------------------------------------
def ldr_resistance_kohm(lux_val, r_dark_kohm=500.0, r_100lux_kohm=5.0, gamma=0.82):
    if lux_val <= 0.0:
        return float(r_dark_kohm)
    r_lit = r_100lux_kohm * ((100.0 / lux_val) ** gamma)
    return float(1.0 / ((1.0 / r_dark_kohm) + (1.0 / r_lit)))

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Variable Light Source")
    source_mode = st.selectbox(
        "Illumination Control Method",
        [
            "Variable Intensity Light Source (0 to 1000 lux)",
            "Polarized Diode Laser + Analyzer (Malus's Law)",
        ],
        index=0,
    )

    if source_mode.startswith("Variable"):
        lux_val = st.slider("Incident Light Intensity $E$ (lux)", 0.0, 1000.0, 200.0, 10.0)
    else:
        e0_lux = st.slider("Unattenuated Laser Intensity $E_0$ (lux)", 100.0, 1000.0, 600.0, 50.0)
        theta_deg = st.slider("Analyzer Angle $\\theta$ (degrees)", 0, 90, 30, 5)
        lux_val = e0_lux * (np.cos(np.deg2rad(theta_deg)) ** 2)
        st.caption(f"Transmitted laser intensity: $E = E_0 \\cos^2({theta_deg}^\\circ) = {lux_val:.1f}$ lux")

    st.subheader("2. Circuit & LDR Parameters")
    vbias_v = st.slider("LDR Bias Voltage $V_{\\text{bias}}$ (V)", 0.5, 5.0, 2.0, 0.1)
    rf_kohm = st.slider(
        "Feedback Resistor $R_f$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Typical feedback resistor is 10 kOhm (adjustable from 1 kOhm to 1 MOhm)."
    )
    r_dark_kohm = st.slider("LDR Dark Resistance $R_{\\text{dark}}$ (kOhm)", 100.0, 1000.0, 500.0, 50.0)
    rcom_kohm = st.slider("Offset Compensation Resistor $R_{\\text{com}}$ (kOhm)", 1.0, 50.0, 10.0, 1.0)
    vcc = st.slider("Dual Power Supply $\\pm V$ (V)", 9.0, 15.0, 12.0, 1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Equations 1 & 2)
# -------------------------------------------------------------------------
vsat = max(vcc - 1.2, 1.0)
rsen_kohm = ldr_resistance_kohm(lux_val, r_dark_kohm=r_dark_kohm)

# Photocurrent through LDR into virtual ground: I = Vbias / Rsen
i_ldr_ma = vbias_v / rsen_kohm

# Equation (1): VO = -Vbias * (Rf / Rsen)
vo_theo = -vbias_v * (rf_kohm / rsen_kohm)
vo_meas = float(np.clip(vo_theo, -vsat, vsat))
saturated = abs(vo_theo) > vsat

# Dark output voltage and Equation (2) signal voltage: Vsig = VO - VO(dark)
vo_dark_theo = -vbias_v * (rf_kohm / r_dark_kohm)
vsig_theo = vo_theo - vo_dark_theo
vsig_meas = vo_meas - float(np.clip(vo_dark_theo, -vsat, vsat))

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LDR Resistance ($R_{\\text{sen}}$)", f"{rsen_kohm:.2f} kOhm")
    m2.metric("Output Voltage ($V_O$)", f"{vo_meas:+.3f} V")
    m3.metric("Dark Output ($V_{O,\\text{dark}}$)", f"{vo_dark_theo:+.3f} V")
    m4.metric("Signal Voltage ($V_{\\text{sig}}$)", f"{vsig_meas:+.3f} V")

    if saturated:
        st.warning(
            f"**Op-Amp Output Saturated ($|V_{{O,\\text{{theo}}}}| = {abs(vo_theo):.2f}$ V > $V_{{sat}} = {vsat:.1f}$ V):** "
            f"Output clipped at **{vo_meas:+.2f} V**. Reduce feedback resistor $R_f$ or bias voltage $V_{{bias}}$ to prevent saturation."
        )
    else:
        st.success(
            f"**Linear Transimpedance Operation ($V_O = {vo_meas:+.3f}$ V):** "
            f"$I_{{LDR}} = \\frac{{V_{{bias}}}}{{R_{{sen}}}} = {i_ldr_ma:.3f}$ mA $\\implies$ "
            f"$V_O = -V_{{bias}} \\frac{{R_f}}{{R_{{sen}}}} = {vo_meas:+.3f}$ V | "
            f"Dark-subtracted signal $V_{{sig}} = {vsig_meas:+.3f}$ V."
        )

    schem_col, plot_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (LM358 Transimpedance Mode)**")
        fig_sch = draw_ldr_tia_schematic(vbias_v, rsen_kohm, rf_kohm, rcom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Eq. (1):** $V_O = -V_{{bias}}\\frac{{R_f}}{{R_{{sen}}}}$; "
            f"**Eq. (2):** $V_{{sig}} = -V_{{bias}} R_f \\left(\\frac{{1}}{{R_{{sen}}}} - \\frac{{1}}{{R_{{dark}}}}\\right)$."
        )

    with plot_col:
        st.markdown("**LDR Resistance & Output Voltage vs. Light Intensity**")
        lux_sweep = np.linspace(0.0, 1000.0, 300)
        rsen_sweep = np.array([ldr_resistance_kohm(e, r_dark_kohm=r_dark_kohm) for e in lux_sweep])
        vo_sweep = np.clip(-vbias_v * (rf_kohm / rsen_sweep), -vsat, vsat)

        fig_ldr, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)

        # Top Plot: LDR Resistance Rsen vs Light Intensity
        ax_r.plot(lux_sweep, rsen_sweep, color="#2ca02c", linewidth=2.2, label="LDR Resistance $R_{sen}$ (kOhm)")
        ax_r.scatter([lux_val], [rsen_kohm], color="#d62728", zorder=6, s=50, label=f"Current ($E={lux_val:.0f}$ lux, $R_{{sen}}={rsen_kohm:.1f}$ k$\\Omega$)")
        ax_r.set_yscale("log")
        style_cro_axes(ax_r, title="Photoconductor Characteristic: LDR Resistance Rsen vs. Light Intensity", xlabel="", ylabel="Rsen (kOhm, log)")
        ax_r.legend(loc="upper right", fontsize=7.5)

        # Bottom Plot: Output Voltage VO vs Light Intensity for multiple Rf values
        for rf_opt, col_opt in [(5.0, "#ff7f0e"), (rf_kohm, "#1f77b4"), (22.0, "#9467bd")]:
            vo_opt = np.clip(-vbias_v * (rf_opt / rsen_sweep), -vsat, vsat)
            lbl = f"$R_f = {rf_opt:.0f}$ k$\\Omega$ (Selected)" if np.isclose(rf_opt, rf_kohm) else f"$R_f = {rf_opt:.0f}$ k$\\Omega$"
            lw = 2.4 if np.isclose(rf_opt, rf_kohm) else 1.5
            ls = "-" if np.isclose(rf_opt, rf_kohm) else "--"
            ax_v.plot(lux_sweep, vo_opt, color=col_opt, linewidth=lw, linestyle=ls, label=lbl)

        ax_v.scatter([lux_val], [vo_meas], color="#d62728", zorder=6, s=50, label=f"Measured $V_O = {vo_meas:+.2f}$ V")
        style_cro_axes(ax_v, title="Transimpedance Output Voltage VO vs. Light Intensity (Sensitivity Comparison)", xlabel="Incident Light Intensity E (lux)", ylabel="Output Voltage VO (V)")
        ax_v.legend(loc="lower left", fontsize=7.3)

        plt.tight_layout()
        st.pyplot(fig_ldr)
        plt.close(fig_ldr)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Exact Lab Observation Table + Aim, Theory & Procedure
    # ---------------------------------------------------------------------
    tab_obs, tab_manual = st.tabs([
        "Observation Table (Dark to Very High Intensity)",
        "Aim, Components, Theory & Procedure"
    ])

    with tab_obs:
        st.markdown(
            f"**Tabular Column ($V_{{bias}} = {vbias_v:.1f}$ V, $R_f = {rf_kohm:.1f}$ k$\\Omega$, $\\pm V = \\pm {vcc:.0f}$ V):**"
        )
        intensity_levels = [
            ("Dark", 0.0),
            ("Low", 25.0),
            ("Medium", 120.0),
            ("High", 400.0),
            ("Very high", 850.0),
        ]
        # Deterministic realistic DMM measurement jitter (~0.5%) for realistic lab comparison
        rng = np.random.default_rng(42)
        obs_rows = []
        for label_int, lux_l in intensity_levels:
            r_s = ldr_resistance_kohm(lux_l, r_dark_kohm=r_dark_kohm)
            v_th = -vbias_v * (rf_kohm / r_s)
            jitter = 1.0 + rng.uniform(-0.006, 0.006)
            v_ms = float(np.clip(v_th * jitter, -vsat, vsat))
            pct_err = abs((v_ms - v_th) / v_th) * 100.0 if abs(v_th) > 1e-9 else 0.0
            status_note = " (Saturated)" if abs(v_th) > vsat else ""
            obs_rows.append({
                "Light Intensity": f"{label_int} ({lux_l:.0f} lux)",
                "LDR Resistance, Rsen (kOhm)": f"{r_s:.2f}",
                "Feedback Resistance Rf (kOhm)": f"{rf_kohm:.1f}",
                "Measured Output Voltage (V)": f"{v_ms:+.3f}{status_note}",
                "Theoretically Calculated Output Voltage (V)": f"{v_th:+.3f}",
                "% Error": f"{pct_err:.2f} %",
            })
        st.table(obs_rows)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Equations

            * **Components:** Operational Amplifier IC (**LM358**), Light Dependent Resistor (**LDR**), Feedback resistor $R_f$ ($1\\text{{ k}}\\Omega\\text{{--}}1\\text{{ M}}\\Omega$, typical $10\\text{{ k}}\\Omega$), Offset compensation resistor $R_{{com}}$ ($10\\text{{ k}}\\Omega$ typical), Dual Power Supply ($\\pm 12\\text{{ V}}$ or $\\pm 15\\text{{ V}}$), Regulated DC Power Supply ($V_{{bias}}$), Digital Multimeter (DMM), Variable intensity light source or polarized diode laser with analyzer.

            * **Theory & Working Equations:**
              1. **Virtual Ground & Current Balance:**
                 Because the non-inverting input (`+`) is referenced to ground via $R_{{com}}$, the inverting input (`-`) is at **virtual ground** ($V_- \\approx 0\\text{{ V}}$). Hence the current flowing through the LDR equals the feedback current through $R_f$:
                 $$I_{{LDR}} = \\frac{{V_{{bias}}}}{{R_{{sen}}}} = I_f = -\\frac{{V_O}}{{R_f}}$$
              2. **Output Voltage (*Equation 1*):**
                 $$V_O = -V_{{bias}} \\left(\\frac{{R_f}}{{R_{{sen}}}}\\right)$$
              3. **Dark-Subtracted Signal Voltage (*Equation 2*):**
                 Subtracting the dark output voltage $V_{{O(\\text{{dark}})}} = -V_{{bias}} \\frac{{R_f}}{{R_{{dark}}}}$ gives the net optical signal voltage:
                 $$V_{{sig}} = V_O - V_{{O(\\text{{dark}})}} = -V_{{bias}} R_f \\left(\\frac{{1}}{{R_{{sen}}}} - \\frac{{1}}{{R_{{dark}}}}\\right)$$
              4. **Design Constraints:**
                 * Choose $R_f$ so $|V_O| < V_{{sat}}$ at maximum expected light intensity to prevent op-amp saturation.
                 * Keep $V_{{bias}}$ below the maximum rated common-mode input voltage of the op-amp.

            * **Procedure:**
              1. Connect the circuit as shown in the schematic diagram.
              2. Measure the LDR resistance $R_{{sen}}$ across the LDR using a digital multimeter (DMM).
              3. Apply dual supply ($+V$ and $-V$) to the LM358 op-amp and bias voltage $V_{{bias}}$ to the LDR.
              4. Keep the LDR in complete darkness and record the dark resistance $R_{{dark}}$ and output voltage $V_{{O(\\text{{dark}})}}$.
              5. Gradually increase the incident light intensity (from `Low` to `Very high`) and record $R_{{sen}}$ and the measured output voltage $V_O$ in each step.
              6. Repeat the observations for different values of feedback resistor $R_f$ to study sensitivity and saturation limits.
            """
        )