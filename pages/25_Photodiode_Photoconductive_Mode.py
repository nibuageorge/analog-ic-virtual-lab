import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 25: Photodiode in Photoconductive Mode", layout="wide")
render_sidebar_header()

st.title("Experiment 25: Silicon PIN Photodiode (BPW34) in Photoconductive Mode")
st.markdown(
    "**Aim:** To study the characteristics and performance of a Silicon PIN Photodiode (**BPW34**) operated in "
    "**Photoconductive (Reverse-Bias) Mode** using an operational amplifier (**LM358**) in transimpedance configuration, "
    "and to observe the variation of output voltage ($V_O$) with incident light intensity."
)

# -------------------------------------------------------------------------
# Helper: Draw Photoconductive (Reverse-Bias) Transimpedance Schematic
# -------------------------------------------------------------------------
def draw_photoconductive_schematic(rf_kohm, vbias_v, vcc):
    with schemdraw.Drawing(show=False, bgcolor="white") as d:
        d.config(unit=2.5, fontsize=8.5, color="#0f172a", lw=1.6)

        # Op-Amp (LM358): Inverting (-) at top-left, Non-Inverting (+) at bottom-left
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label("LM358", loc="center", fontsize=8.5))
        d.add(elm.Line().at(op.vd).to((op.vd[0], op.vd[1] + 0.35)).label(f"+{vcc:.0f}V", loc="top", fontsize=7.8))
        d.add(elm.Line().at(op.vs).to((op.vs[0], op.vs[1] - 0.35)).label(f"-{vcc:.0f}V", loc="bottom", fontsize=7.8))

        # Output node Vo
        out_x, out_y = op.out
        fb_x = out_x + 0.85
        d.add(elm.Line().at((out_x, out_y)).to((fb_x, out_y)))
        d.add(elm.Dot().at((fb_x, out_y)))
        d.add(elm.Line().at((fb_x, out_y)).to((fb_x + 0.85, out_y)).label("$V_O$", loc="right", fontsize=9.5))

        # Inverting (-) virtual-ground summing node
        inv_x, inv_y = op.in1
        sx = inv_x - 0.95
        sy = inv_y
        d.add(elm.Line().at((inv_x, inv_y)).to((sx, sy)))
        d.add(elm.Dot().at((sx, sy)))

        # Feedback Resistor Rf above Op-Amp
        top_y = sy + 1.35
        d.add(elm.Line().at((sx, sy)).to((sx, top_y)))
        d.add(elm.Resistor().at((sx, top_y)).to((fb_x, top_y)).label(f"$R_f$\n{rf_kohm:.0f} k$\\Omega$", loc="top", fontsize=8.5))
        d.add(elm.Line().at((fb_x, top_y)).to((fb_x, out_y)))

        # Non-Inverting (+) input connected directly to Ground
        noninv_x, noninv_y = op.in2
        gnd_plus_x = noninv_x - 0.55
        d.add(elm.Line().at((noninv_x, noninv_y)).to((gnd_plus_x, noninv_y)))
        d.add(elm.Ground().at((gnd_plus_x, noninv_y)))

        # Reverse-Bias Section (Left of virtual ground):
        # Cathode on left (connected to +Vbias), Anode on right (connected to inverting '-' node)
        src_x = sx - 2.35
        gnd_bias_y = sy - 1.85

        d.add(
            elm.Photodiode()
            .right()
            .at((src_x, sy))
            .tox(sx)
            .label("BPW34 (Reverse-Bias)", loc="top", fontsize=8.0)
        )
        d.add(elm.Dot().at((src_x, sy)))

        # DC Reverse-Bias Source (+Vbias at top connected to Cathode, '-' grounded at bottom)
        d.add(
            elm.SourceV()
            .up()
            .at((src_x, gnd_bias_y))
            .toy(sy)
            .label(f"$V_{{bias}}$\n+{vbias_v:.1f} V", loc="left", fontsize=8.2)
        )
        d.add(elm.Ground().at((src_x, gnd_bias_y)))

        buf = io.BytesIO()
        d.save(buf, dpi=220, transparent=False)
        buf.seek(0)
        plt.close("all")
        return buf

# -------------------------------------------------------------------------
# Physics Model: BPW34 in Photoconductive (Reverse-Bias) Mode
# -------------------------------------------------------------------------
def compute_photoconductive_response(irradiance_wm2, rf_kohm, vbias_v, vcc):
    vsat = max(vcc - 1.2, 1.0)
    area_m2 = 7.5e-6  # 7.5 mm^2 active radiant sensitive area of BPW34
    responsivity_aw = 0.62  # A/W peak responsivity

    p_opt_uw = irradiance_wm2 * area_m2 * 1e6
    i_ph_ua = responsivity_aw * p_opt_uw
    i_dark_ua = 0.002 * (1.0 + 0.15 * vbias_v)
    i_total_ua = i_ph_ua + i_dark_ua

    vo_theo = -(i_total_ua * 1e-6) * (rf_kohm * 1e3)
    vo_meas = float(np.clip(vo_theo, -vsat, 0.0))
    saturated = abs(vo_theo) > vsat

    depletion_width_um = 4.5 * np.sqrt(1.0 + vbias_v / 0.65)
    bandwidth_khz = 35.0 * np.sqrt(1.0 + vbias_v / 0.65)

    return {
        "p_opt_uw": p_opt_uw,
        "i_ph_ua": i_ph_ua,
        "i_dark_ua": i_dark_ua,
        "vo_theo": vo_theo,
        "vo_meas": vo_meas,
        "saturated": saturated,
        "vsat": vsat,
        "depletion_width_um": depletion_width_um,
        "bandwidth_khz": bandwidth_khz,
    }

# -------------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Variable Light Source")
    irrad_wm2 = st.slider(
        "Incident Light Intensity $E$ (W/m$^2$)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
        help="0 W/m^2 corresponds to complete darkness; 200 W/m^2 corresponds to bright illumination."
    )

    st.subheader("2. Reverse Bias & Transimpedance Gain")
    vbias_v = st.slider(
        "Reverse Bias Voltage $+V_{\\text{bias}}$ (V)",
        min_value=1.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        help="Applied to the cathode of BPW34 to widen the depletion region."
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_f$ (kOhm)",
        min_value=100.0,
        max_value=1000.0,
        value=220.0,
        step=20.0,
        help="Standard feedback resistor range for photoconductive mode: 100 kOhm to 1 MOhm."
    )
    vcc = st.slider("Dual Power Supply $\\pm V$ (V)", 9.0, 15.0, 12.0, 1.0)

res = compute_photoconductive_response(irrad_wm2, rf_kohm, vbias_v, vcc)

with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optical Power ($P_{\\text{opt}}$)", f"{res['p_opt_uw']:.2f} uW")
    m2.metric("Photocurrent ($I_{ph}$)", f"{res['i_ph_ua']:.2f} uA")
    m3.metric("Output Voltage ($V_O$)", f"{res['vo_meas']:+.3f} V")
    m4.metric("Dark Current ($I_{\\text{dark}}$)", f"{res['i_dark_ua']*1e3:.1f} nA")

    if res["saturated"]:
        st.warning(
            f"**Op-Amp Saturated ($|V_{{O,\\text{{theo}}}}| = {abs(res['vo_theo']):.2f}$ V > $V_{{sat}} = {res['vsat']:.1f}$ V):** "
            f"Output clipped at **{res['vo_meas']:+.2f} V**. Decrease $R_f$ or light intensity."
        )
    else:
        st.success(
            f"**Linear Photoconductive Operation ($V_O = -I_{{ph}} R_f$):** "
            f"At $E = {irrad_wm2:.0f}$ W/m$^2$ ($I_{{ph}} = {res['i_ph_ua']:.2f}\\;\\mu$A, $R_f = {rf_kohm:.0f}$ k$\\Omega$), "
            f"measured **$V_O = {res['vo_meas']:+.3f}$ V** (Depletion width $\\approx {res['depletion_width_um']:.1f}\\;\\mu$m, Bandwidth $\\approx {res['bandwidth_khz']:.0f}$ kHz)."
        )

    schem_col, plot_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram: Photoconductive Mode**")
        sch_buf = draw_photoconductive_schematic(rf_kohm, vbias_v, vcc)
        st.image(sch_buf, use_container_width=True)
        st.caption(f"**Reverse-Bias Mode:** Cathode at $+{vbias_v:.1f}\\text{{ V}}$, anode at virtual ground ($V_O = -I_{{ph}} R_f$).")

    with plot_col:
        st.markdown("**Light Intensity vs. Output Voltage ($V_O$)**")
        irrad_sweep = np.linspace(0.0, 200.0, 300)
        fig_pd, ax_pd = plt.subplots(figsize=(7.6, 4.8))

        rf_list = sorted(list({100.0, rf_kohm, 500.0}))
        colors_rf = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
        for r_opt, col_opt in zip(rf_list, colors_rf):
            vo_curve = [compute_photoconductive_response(e, r_opt, vbias_v, vcc)["vo_meas"] for e in irrad_sweep]
            is_sel = np.isclose(r_opt, rf_kohm)
            ax_pd.plot(
                irrad_sweep,
                vo_curve,
                color=col_opt,
                linewidth=2.4 if is_sel else 1.5,
                linestyle="-" if is_sel else "--",
                label=f"$R_f = {r_opt:.0f}$ k$\\Omega$" + (" (Selected)" if is_sel else "")
            )

        ax_pd.scatter([irrad_wm2], [res["vo_meas"]], color="#d62728", zorder=6, s=55, label=f"Operating Point ({irrad_wm2:.0f} W/m$^2$, {res['vo_meas']:+.2f} V)")
        style_cro_axes(
            ax_pd,
            title="Photoconductive Mode: Output Voltage VO vs. Light Intensity",
            xlabel="Light Intensity E (W/m^2)",
            ylabel="Output Voltage VO (V)"
        )
        ax_pd.legend(loc="lower left", fontsize=7.6)
        plt.tight_layout()
        st.pyplot(fig_pd, clear_figure=True)
        plt.close(fig_pd)

    tab_obs, tab_manual = st.tabs([
        "Observation Table (Light Intensity vs. Output Voltage)",
        "Aim, Components, Theory & Procedure"
    ])

    with tab_obs:
        st.markdown(f"**Tabular Column ($R_f = {rf_kohm:.0f}$ k$\\Omega$, $+V_{{bias}} = +{vbias_v:.1f}$ V, $\\pm V = \\pm {vcc:.0f}$ V):**")
        e_steps = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 200.0]
        rows = []
        for e_val in e_steps:
            r_e = compute_photoconductive_response(e_val, rf_kohm, vbias_v, vcc)
            rows.append({
                "Light Intensity": "0 (Dark)" if e_val == 0 else f"{e_val:.0f} W/m^2",
                "Output Voltage (V)": f"{r_e['vo_meas']:+.3f}" + (" [Saturated]" if r_e["saturated"] else ""),
            })
        st.table(rows)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up

            * **Components:** Photodiode (Silicon PIN photodiode - **BPW34**), Op-amp (**LM358**), Feedback Resistor $R_f$ ($100\\text{{ k}}\\Omega\\text{{--}}1\\text{{ M}}\\Omega$), Power Supply ($\\pm 12\\text{{ V}}$ / $\\pm 15\\text{{ V}}$ DC), Regulated Bias Supply ($+V_{{bias}}$), Variable Light Source, Digital Multimeter, Breadboard & Wires.

            * **Theory:**
              When operated in **photoconductive mode**, the photodiode is reverse biased, which increases the width of the depletion region and improves:
              * Response speed (lower junction capacitance)
              * Linearity across higher light levels
              * Sensitivity

              The photodiode cathode is connected to $+V_{{bias}}$ and the anode is connected to the inverting (`-`) virtual-ground node of the op-amp. The photocurrent flows through $R_f$, producing:
              $$V_O = -I_{{ph}} R_f$$

            * **Comment on Graph Shape:**
              The plot of **Light Intensity versus Output Voltage ($V_O$)** is a **linear straight line** with negative slope ($-\\mathcal{{R}} A R_f$), confirming direct proportionality between output voltage and incident light intensity until op-amp saturation ($-V_{{sat}} \\approx -{res['vsat']:.1f}\\text{{ V}}$).
            """
        )
