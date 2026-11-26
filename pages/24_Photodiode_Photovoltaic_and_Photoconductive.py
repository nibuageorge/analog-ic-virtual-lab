import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 24: Photodiode - Photovoltaic & Photoconductive Modes", layout="wide")
render_sidebar_header()

st.title("Experiment 24: Silicon PIN Photodiode (BPW34) -- Photovoltaic & Photoconductive Modes")
st.markdown(
    "**Aim:** To study and compare the performance characteristics of a Silicon PIN Photodiode (**BPW34**) "
    "operated in **Part I: Photovoltaic (Zero-Bias) Mode** and **Part II: Photoconductive (Reverse-Bias) Mode** "
    "using an operational amplifier (**LM358**) in transimpedance configuration."
)

# -------------------------------------------------------------------------
# Helper: Draw Clean, Explicit-Canvas Schematics for Both Photodiode Modes
# -------------------------------------------------------------------------
def draw_photodiode_schematic(mode_key, rf_kohm, vbias_v, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.2)

        # Op-Amp (LM358): Inverting (-) on top, Non-Inverting (+) on bottom
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label("LM358", loc="center", fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f"+{vcc:.0f}V", loc="top", fontsize=7.8))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f"-{vcc:.0f}V", loc="bottom", fontsize=7.8))

        # Output terminal Vo
        d.add(elm.Line().right().at(op.out).length(0.75))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center
        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85).label("$V_O$", loc="right", fontsize=9.5))

        # Inverting input (-) virtual-ground summing node
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.85))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # Feedback resistor Rf above op-amp
        top_y = sy + 1.35
        d.add(elm.Line().up().at((sx, sy)).toy(top_y))
        d.add(elm.Resistor().right().tox(fb_x).label(f"$R_f$\n{rf_kohm:.0f} k$\\Omega$", loc="top", fontsize=8.5))
        d.add(elm.Line().down().toy(fb_y))

        # Non-inverting (+) input connected to Ground
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.55))
        d.add(elm.Ground())

        if mode_key == "photovoltaic":
            # Part I: Photovoltaic (Zero-Bias) Mode
            # Anode at Ground, Cathode connected to inverting (-) virtual ground -> Vo = +Iph * Rf
            pd_x = sx - 1.15
            gnd_y = sy - 1.75
            d.add(elm.Line().left().at((sx, sy)).to((pd_x, sy)))
            d.add(
                elm.Photodiode()
                .at((pd_x, gnd_y))
                .to((pd_x, sy))
                .label("BPW34\n(Zero-Bias)", loc="left", fontsize=8.2)
            )
            d.add(elm.Ground().at((pd_x, gnd_y)))
        else:
            # Part II: Photoconductive (Reverse-Bias) Mode
            # Cathode connected to +Vbias, Anode connected to inverting (-) virtual ground -> Vo = -Iph * Rf
            src_x = sx - 2.15
            d.add(
                elm.Photodiode()
                .at((sx, sy))
                .to((src_x, sy))
                .label("BPW34\n(Reverse-Bias)", loc="top", fontsize=8.2)
            )
            d.add(elm.Line().down().at((src_x, sy)).length(0.2))
            d.add(elm.SourceV().down().length(1.25).label(f"$+V_{{bias}}$\n{vbias_v:.1f} V", loc="left", fontsize=8.2))
            d.add(elm.Ground())

        fig = d.draw(show=False).fig
        return fig
# -------------------------------------------------------------------------
# Physics Model: BPW34 Silicon PIN Photodiode
# Active area A = 7.5 mm^2 = 0.075 cm^2; Peak Responsivity R_lambda ~ 0.62 A/W
# -------------------------------------------------------------------------
def compute_bpw34_response(mode_key, irradiance_wm2, rf_kohm, vbias_v, vcc):
    vsat = max(vcc - 1.2, 1.0)
    area_m2 = 7.5e-6  # 7.5 mm^2 active radiant sensitive area of BPW34
    responsivity_aw = 0.62  # A/W at visible/NIR peak

    p_opt_uw = irradiance_wm2 * area_m2 * 1e6  # incident optical power in uW
    i_ph_ua = responsivity_aw * p_opt_uw  # photocurrent in uA

    if mode_key == "photovoltaic":
        i_dark_ua = 0.0
        i_total_ua = i_ph_ua
        vo_theo = +(i_total_ua * 1e-6) * (rf_kohm * 1e3)
        depletion_width_um = 4.5
        bandwidth_khz = 35.0
    else:
        i_dark_ua = 0.002 * (1.0 + 0.15 * vbias_v)
        i_total_ua = i_ph_ua + i_dark_ua
        vo_theo = -(i_total_ua * 1e-6) * (rf_kohm * 1e3)
        depletion_width_um = 4.5 * np.sqrt(1.0 + vbias_v / 0.65)
        bandwidth_khz = 35.0 * np.sqrt(1.0 + vbias_v / 0.65)

    vo_meas = float(np.clip(vo_theo, -vsat, vsat))
    saturated = abs(vo_theo) > vsat

    return {
        "p_opt_uw": p_opt_uw,
        "i_ph_ua": i_ph_ua,
        "i_dark_ua": i_dark_ua,
        "i_total_ua": i_total_ua,
        "vo_theo": vo_theo,
        "vo_meas": vo_meas,
        "saturated": saturated,
        "vsat": vsat,
        "depletion_width_um": depletion_width_um,
        "bandwidth_khz": bandwidth_khz,
    }

# -------------------------------------------------------------------------
# Top Mode Switcher: Part I (Photovoltaic) vs Part II (Photoconductive)
# -------------------------------------------------------------------------
mode_choice = st.radio(
    "Select Photodiode Operating Mode",
    [
        "Part I: Photovoltaic Mode (Zero Bias, VR = 0 V)",
        "Part II: Photoconductive Mode (Reverse Bias, Cathode at +Vbias)",
    ],
    horizontal=True,
    key="bpw34_mode_selector",
)
mode_key = "photovoltaic" if mode_choice.startswith("Part I") else "photoconductive"
st.divider()

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Variable Light Source")
    irrad_wm2 = st.slider(
        "Incident Light Intensity / Irradiance $E$ (W/m$^2$)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
        help="0 W/m^2 corresponds to complete darkness; 200 W/m^2 corresponds to bright illumination."
    )

    st.subheader("2. Transimpedance & Bias Settings")
    default_rf = 100.0 if mode_key == "photovoltaic" else 220.0
    rf_kohm = st.slider(
        "Feedback Resistor $R_f$ (kOhm)",
        min_value=10.0,
        max_value=1000.0,
        value=default_rf,
        step=10.0,
        help="Photovoltaic mode typically uses 10 kOhm - 1 MOhm; Photoconductive mode typically uses 100 kOhm - 1 MOhm."
    )

    vbias_v = st.slider(
        "Reverse Bias Voltage $+V_{\\text{bias}}$ (V)",
        min_value=1.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        disabled=(mode_key == "photovoltaic"),
        help="Active only in Photoconductive Mode (zero external bias in Photovoltaic Mode)."
    )

    vcc = st.slider("Dual Power Supply $\\pm V$ (V)", 9.0, 15.0, 12.0, 1.0)

# -------------------------------------------------------------------------
# Compute Operating State
# -------------------------------------------------------------------------
res = compute_bpw34_response(mode_key, irrad_wm2, rf_kohm, vbias_v, vcc)

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optical Power ($P_{\\text{opt}}$)", f"{res['p_opt_uw']:.2f} uW")
    m2.metric("Photocurrent ($I_{ph}$)", f"{res['i_ph_ua']:.2f} uA")
    m3.metric("Output Voltage ($V_O$)", f"{res['vo_meas']:+.3f} V")
    m4.metric("Dark Current ($I_{\\text{dark}}$)", f"{res['i_dark_ua']*1e3:.1f} nA")

    if res["saturated"]:
        st.warning(
            f"**Op-Amp Output Saturated ($|V_{{O,\\text{{theo}}}}| = {abs(res['vo_theo']):.2f}$ V > $V_{{sat}} = {res['vsat']:.1f}$ V):** "
            f"Output clipped at **{res['vo_meas']:+.2f} V**. Reduce feedback resistor $R_f$ or incident light intensity."
        )
    else:
        sign_note = "$V_O = +I_{ph} R_f$" if mode_key == "photovoltaic" else "$V_O = -(I_{ph} + I_{\\text{dark}}) R_f$"
        st.success(
            f"**Linear Transimpedance Response ({sign_note}):** "
            f"At $E = {irrad_wm2:.0f}$ W/m$^2$ ($I_{{ph}} = {res['i_ph_ua']:.2f}\\;\\mu$A, $R_f = {rf_kohm:.0f}$ k$\\Omega$), "
            f"measured **$V_O = {res['vo_meas']:+.3f}$ V** (Depletion width $\\approx {res['depletion_width_um']:.1f}\\;\\mu$m, Bandwidth $\\approx {res['bandwidth_khz']:.0f}$ kHz)."
        )

    schem_col, plot_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        mode_short = "Photovoltaic (Zero-Bias)" if mode_key == "photovoltaic" else "Photoconductive (Reverse-Bias)"
        st.markdown(f"**Circuit Diagram: {mode_short}**")
        fig_sch = draw_photodiode_schematic(mode_key, rf_kohm, vbias_v, vcc)
        st.pyplot(fig_sch, clear_figure=True)
        plt.close(fig_sch)
        if mode_key == "photovoltaic":
            st.caption("**Photovoltaic Mode:** $V_R = 0\\text{ V}$, zero dark current ($I_{\\text{dark}} = 0$), $V_O = I_{ph} R_f$.")
        else:
            st.caption(f"**Photoconductive Mode:** Cathode at $+{vbias_v:.1f}\\text{{ V}}$, anode at virtual ground, $V_O = -I_{{ph}} R_f$.")

    with plot_col:
        st.markdown("**Light Intensity vs. Output Voltage ($V_O$) Characteristic**")
        irrad_sweep = np.linspace(0.0, 200.0, 300)

        fig_pd, ax_pd = plt.subplots(figsize=(7.6, 4.9))
        rf_list = sorted(list({50.0, rf_kohm, 500.0})) if mode_key == "photovoltaic" else sorted(list({100.0, rf_kohm, 500.0}))
        colors_rf = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]

        for r_opt, col_opt in zip(rf_list, colors_rf):
            vo_curve = [compute_bpw34_response(mode_key, e, r_opt, vbias_v, vcc)["vo_meas"] for e in irrad_sweep]
            is_sel = np.isclose(r_opt, rf_kohm)
            ax_pd.plot(
                irrad_sweep,
                vo_curve,
                color=col_opt,
                linewidth=2.4 if is_sel else 1.5,
                linestyle="-" if is_sel else "--",
                label=f"$R_f = {r_opt:.0f}$ k$\\Omega$" + (" (Selected)" if is_sel else "")
            )

        ax_pd.scatter([irrad_wm2], [res["vo_meas"]], color="#d62728", zorder=6, s=55, label=f"Measured ($E={irrad_wm2:.0f}$ W/m$^2$, $V_O={res['vo_meas']:+.2f}$ V)")
        style_cro_axes(
            ax_pd,
            title=f"BPW34 {mode_short}: Output Voltage VO vs. Incident Light Intensity",
            xlabel="Incident Light Intensity E (W/m^2)",
            ylabel="Output Voltage VO (V)"
        )
        ax_pd.legend(loc="best", fontsize=7.6)
        plt.tight_layout()
        st.pyplot(fig_pd, clear_figure=True)
        plt.close(fig_pd)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Observation Table + Mode Comparison + Lab Manual
    # ---------------------------------------------------------------------
    tab_obs, tab_cmp, tab_manual = st.tabs([
        "Observation Table (Light Intensity vs. Output Voltage)",
        "Mode Comparison (Photovoltaic vs. Photoconductive)",
        "Aim, Components, Theory & Procedure"
    ])

    with tab_obs:
        st.markdown(
            f"**Tabular Column -- {mode_short} ($R_f = {rf_kohm:.0f}$ k$\\Omega$, $\\pm V = \\pm {vcc:.0f}$ V):**"
        )
        e_steps = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 200.0]
        obs_rows = []
        for e_val in e_steps:
            r_e = compute_bpw34_response(mode_key, e_val, rf_kohm, vbias_v, vcc)
            obs_rows.append({
                "Light Intensity E (W/m^2)": "0 (Dark)" if e_val == 0 else f"{e_val:.0f}",
                "Incident Optical Power (uW)": f"{r_e['p_opt_uw']:.2f}",
                "Photocurrent Iph (uA)": f"{r_e['i_ph_ua']:.2f}",
                "Output Voltage VO (V)": f"{r_e['vo_meas']:+.3f}" + (" [Saturated]" if r_e["saturated"] else ""),
            })
        st.table(obs_rows)

    with tab_cmp:
        st.markdown(
            f"Direct magnitude comparison ($|V_O|$) between **Photovoltaic (Zero-Bias)** and **Photoconductive (Reverse-Bias = {vbias_v:.1f} V)** "
            f"at **$R_f = {rf_kohm:.0f}$ k$\\Omega$**:"
        )
        irrad_cmp = np.linspace(0.0, 200.0, 250)
        vo_pv_abs = [abs(compute_bpw34_response("photovoltaic", e, rf_kohm, 0.0, vcc)["vo_meas"]) for e in irrad_cmp]
        vo_pc_abs = [abs(compute_bpw34_response("photoconductive", e, rf_kohm, vbias_v, vcc)["vo_meas"]) for e in irrad_cmp]

        fig_c, ax_c = plt.subplots(figsize=(8.6, 3.6))
        ax_c.plot(irrad_cmp, vo_pv_abs, color="#1f77b4", linewidth=2.2, label="Part I: Photovoltaic Mode ($V_R = 0$ V, $I_{dark} = 0$ nA)")
        ax_c.plot(irrad_cmp, vo_pc_abs, color="#d62728", linestyle="--", linewidth=2.0, label=f"Part II: Photoconductive Mode ($V_{{bias}} = +{vbias_v:.1f}$ V, wider depletion & faster)")
        style_cro_axes(ax_c, title="Linearity Comparison: |VO| vs. Incident Light Intensity", xlabel="Light Intensity E (W/m^2)", ylabel="|VO| (V)")
        ax_c.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_c, clear_figure=True)
        plt.close(fig_c)

        st.markdown(
            """
            | Feature | Part I: Photovoltaic Mode | Part II: Photoconductive Mode |
            | :--- | :--- | :--- |
            | **External Bias ($V_R$)** | **Zero ($0\\text{ V}$)** across diode | **Reverse Bias ($+V_{\\text{bias}}$)** on cathode |
            | **Dark Current ($I_{\\text{dark}}$)** | Virtually zero ($0\\text{ nA}$) $\\rightarrow$ lowest noise | Small reverse leakage ($\\sim 2\\text{--}5\\text{ nA}$) |
            | **Depletion Width & Speed** | Narrower junction $\\rightarrow$ lower bandwidth (low-speed / precision) | Wider depletion region $\\rightarrow$ fast response speed & high linearity |
            | **Output Voltage Formula** | $V_O = +I_{ph} R_f$ | $V_O = -(I_{ph} + I_{\\text{dark}}) R_f$ |
            """
        )

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Observation Analysis

            * **Components:** Silicon PIN Photodiode (**BPW34**), Op-Amp (**LM358**), Feedback Resistor $R_f$ ($10\\text{{ k}}\\Omega\\text{{--}}1\\text{{ M}}\\Omega$), Dual Power Supply ($\\pm 12\\text{{ V}}$ or $\\pm 15\\text{{ V}}$), Regulated DC Supply ($+V_{{bias}}$ for Part II), Variable Light Source, Digital Multimeter (DMM).

            * **Theory:**
              1. **Part I -- Photovoltaic (Zero-Bias) Mode:**
                 The photodiode is connected with zero external bias across the op-amp inputs (anode grounded, cathode at inverting virtual ground). Incident photons generate electron-hole pairs swept by the built-in junction field, producing photocurrent $I_{{ph}}$ proportional to incident light intensity:
                 $$V_O = I_{{ph}} R_f$$
                 Because there is no external bias, dark current is eliminated, yielding low-noise operation ideal for precision low-light measurements.
              2. **Part II -- Photoconductive (Reverse-Bias) Mode:**
                 The photodiode cathode is connected to $+V_{{bias}}$ and the anode to the inverting (`-`) virtual-ground node. Reverse bias widens the depletion region, reducing junction capacitance and improving response speed, linearity, and sensitivity:
                 $$V_O = -I_{{ph}} R_f$$

            * **Shape of Graph & Comment:**
              * The plot of **Output Voltage ($V_O$) versus Light Intensity ($E$)** is a **straight line passing through the origin** (with a tiny dark offset in photoconductive mode), confirming that $V_O$ is **directly proportional to incident light intensity** until the output reaches the op-amp saturation voltage ($\\pm V_{{sat}} \\approx \\pm {res['vsat']:.1f}\\text{{ V}}$).
            """
        )
