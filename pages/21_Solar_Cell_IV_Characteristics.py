import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from scipy.optimize import brentq
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 21: Solar Cell - Efficiency and Fill Factor", layout="wide")
render_sidebar_header()

st.title("Experiment 21: Solar Cell -- Efficiency and Fill Factor")
st.markdown(
    "**Aim:** To plot the $V$-$I$ characteristics of the solar cell under an incandescent lamp and hence determine "
    "the **Fill Factor ($FF$)** and **Efficiency ($\\eta$)**."
)

# -------------------------------------------------------------------------
# Helper: Draw Solar Cell Experimental Schematic
# -------------------------------------------------------------------------
def draw_solar_cell_schematic(r_load_ohm, dist_cm, p_light_w):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        # Left branch: Solar Cell illuminated by incandescent bulb at distance r
        sc = d.add(
            elm.Photodiode()
            .up()
            .at((0, 0))
            .length(2.5)
            .label(f"Solar Cell\n($P_{{light}}={p_light_w:.0f}$ W,\n$r={dist_cm:.0f}$ cm)", loc="left", fontsize=8.0)
        )
        top_y = sc.end[1]
        bot_y = sc.start[1]

        # Top wire: Solar Cell (+) -> Milli-ammeter -> Voltmeter tap -> Variable Load R
        d.add(elm.Line().right().at((0, top_y)).length(0.55))
        d.add(elm.MeterA().right().length(1.4).label("mA", loc="top", fontsize=8.5))
        d.add(elm.Line().right().length(0.55))
        v_top = d.add(elm.Dot())
        vx = v_top.center[0]

        # Voltmeter in parallel across Solar Cell / Load
        d.add(elm.MeterV().down().at((vx, top_y)).toy(bot_y).label("V", loc="left", fontsize=8.5))
        d.add(elm.Dot())

        # Variable Load Resistor R
        d.add(elm.Line().right().at((vx, top_y)).length(1.55))
        rx = d.here[0]
        r_label = "R = inf (Open)" if np.isinf(r_load_ohm) else f"R = {r_load_ohm:.0f} Ohm"
        d.add(elm.ResistorVar().down().at((rx, top_y)).toy(bot_y).label(r_label, loc="right", fontsize=8.0))

        # Bottom return wire to Solar Cell (-)
        d.add(elm.Line().left().at((rx, bot_y)).tox(0))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Physics Model: Eqs. (1)-(3) from Lab Manual
# I = -Iph + I0 * (exp(eV / (eta * k * T)) - 1)
# -------------------------------------------------------------------------
def compute_solar_cell_curve(p_bulb_w, optical_frac, dist_cm, area_cm2, eta_ideality, temp_c):
    e_charge = 1.60217663e-19
    k_boltz = 1.380649e-23
    T_kelvin = temp_c + 273.15
    vt = (eta_ideality * k_boltz * T_kelvin) / e_charge

    # Spherical radiation distribution: I_rad = P_light / (4 * pi * r^2)
    p_light_w = p_bulb_w * optical_frac
    r_m = dist_cm * 1e-2
    area_m2 = area_cm2 * 1e-4

    irradiance_wm2 = p_light_w / (4.0 * np.pi * (r_m ** 2))
    p_in_w = irradiance_wm2 * area_m2
    p_in_mw = p_in_w * 1e3

    # Photocurrent Iph = K * Irradiance (calibrated to typical lab Si solar cell: ~35 mA at 1000 W/m^2 for 10 cm^2)
    k_device = 3.5e-5  # A / (W/m^2) per cm^2
    i_ph = k_device * area_cm2 * irradiance_wm2

    # Reverse saturation current I0 (A)
    i_0 = 2.5e-9 * area_cm2

    # Open-circuit voltage Voc when I = 0 => Iph = I0 * (exp(Voc / vt) - 1)
    voc = vt * np.log((i_ph / i_0) + 1.0) if i_ph > 0 else 0.0

    # Sweep voltage V from 0 to Voc
    v_arr = np.linspace(0.0, voc, 600)
    # Eq. (2): I = -Iph + I0 * (exp(eV / (eta*k*T)) - 1)
    i_fourth_quad = -i_ph + i_0 * (np.exp(np.clip(v_arr / vt, -40, 45)) - 1.0)
    i_mag = np.maximum(-i_fourth_quad, 0.0)
    p_out_w = v_arr * i_mag

    return {
        "p_light_w": p_light_w,
        "irradiance_wm2": irradiance_wm2,
        "p_in_mw": p_in_mw,
        "i_ph": i_ph,
        "i_0": i_0,
        "vt": vt,
        "voc": voc,
        "v_arr": v_arr,
        "i_fourth_quad": i_fourth_quad,
        "i_mag": i_mag,
        "p_out_w": p_out_w,
    }

def solve_for_load_resistance(r_ohm, model):
    i_ph = model["i_ph"]
    i_0 = model["i_0"]
    vt = model["vt"]
    voc = model["voc"]

    if r_ohm == 0.0:
        return 0.0, i_ph, 0.0
    if np.isinf(r_ohm):
        return voc, 0.0, 0.0

    # Intersection of Eq. (2) and Eq. (3): V/R = Iph - I0*(exp(V/vt) - 1)
    def f_op(v):
        return i_ph - i_0 * (np.exp(np.clip(v / vt, -40, 45)) - 1.0) - (v / r_ohm)

    try:
        v_op = brentq(f_op, 0.0, voc)
    except ValueError:
        v_op = 0.0
    i_op = v_op / r_ohm
    p_op_mw = v_op * i_op * 1e3
    return v_op, i_op, p_op_mw

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Lamp & Geometry Setup")
    p_bulb_w = st.slider("Incandescent Bulb Electrical Power (W)", 40.0, 200.0, 100.0, 10.0)
    optical_pct = st.slider(
        "Optical Output Fraction of Bulb (%)",
        min_value=10.0,
        max_value=30.0,
        value=20.0,
        step=1.0,
        help="Lab manual assumption: 100 W incandescent bulb emits ~20 W of light power (20%)."
    )
    dist_cm = st.slider("Bulb-to-Solar-Cell Distance $r$ (cm)", 5.0, 30.0, 10.0, 1.0)
    area_cm2 = st.slider("Solar Cell Surface Area $A$ (cm$^2$)", 2.0, 25.0, 10.0, 0.5)

    st.subheader("2. Variable Load Resistance ($R$)")
    load_presets = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, "Infinity"]
    r_selected = st.select_slider(
        "Select Load Resistor $R$ (Ohm)",
        options=load_presets,
        value=20,
        help="Vary R from 0 Ohm (short circuit, Isc) to Infinity (open circuit, Voc)."
    )
    r_load_ohm = np.inf if r_selected == "Infinity" else float(r_selected)

    st.subheader("3. Diode Parameters")
    eta_ideality = st.slider("Diode Ideality Factor $\\eta$", 1.0, 2.0, 1.35, 0.05)
    temp_c = st.slider("Ambient Temperature $T$ ($^\\circ$C)", 20.0, 45.0, 27.0, 1.0)

# -------------------------------------------------------------------------
# Compute Curves, MPP, Fill Factor & Efficiency
# -------------------------------------------------------------------------
model = compute_solar_cell_curve(p_bulb_w, optical_pct / 100.0, dist_cm, area_cm2, eta_ideality, temp_c)

isc_ma = model["i_ph"] * 1e3
voc_v = model["voc"]

idx_mpp = int(np.argmax(model["p_out_w"]))
vm_v = model["v_arr"][idx_mpp]
im_ma = model["i_mag"][idx_mpp] * 1e3
pmax_mw = model["p_out_w"][idx_mpp] * 1e3
r_opt_ohm = vm_v / max(im_ma * 1e-3, 1e-9)

# Eq. (4): Fill Factor FF = (Im * Vm) / (Isc * Voc)
ff_val = (im_ma * vm_v) / (isc_ma * voc_v) if (isc_ma * voc_v) > 0 else 0.0

# Eq. (5): Efficiency = P_out(max) / P_in
eff_pct = (pmax_mw / model["p_in_mw"]) * 100.0 if model["p_in_mw"] > 0 else 0.0

# Current load operating point (V', I')
v_prime, i_prime_a, p_prime_mw = solve_for_load_resistance(r_load_ohm, model)
i_prime_ma = i_prime_a * 1e3

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Short-Circuit $I_{sc}$ ($R=0$)", f"{isc_ma:.2f} mA")
    m2.metric("Open-Circuit $V_{oc}$ ($R=\\infty$)", f"{voc_v:.3f} V")
    m3.metric("Fill Factor ($FF$)", f"{ff_val:.3f} ({ff_val*100:.1f}%)")
    m4.metric("Efficiency ($\\eta = P_{{out}}/P_{{in}}$)", f"{eff_pct:.2f} %")

    st.info(
        f"**Incident Radiation & Power Balance:** "
        f"$P_{{light}} = {model['p_light_w']:.1f}$ W over sphere of radius $r = {dist_cm:.0f}$ cm $\\implies$ "
        f"Irradiance $= {model['irradiance_wm2']:.2f}$ W/m$^2$ $\\implies$ "
        f"Incident Power $P_{{in}} = {model['p_in_mw']:.2f}$ mW | "
        f"**Max Output $P_{{out}} = {pmax_mw:.2f}$ mW** at $V_m = {vm_v:.3f}$ V, $I_m = {im_ma:.2f}$ mA ($R_{{opt}} \\approx {r_opt_ohm:.1f}\\;\\Omega$)."
    )

    schem_col, plot_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Experimental Circuit Diagram**")
        fig_sch = draw_solar_cell_schematic(r_load_ohm, dist_cm, model["p_light_w"])
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        r_disp = "Infinity" if np.isinf(r_load_ohm) else f"{r_load_ohm:.0f} Ohm"
        st.caption(
            f"**Operating Point P ($R = {r_disp}$):** "
            f"$V' = {v_prime:.3f}$ V, $I' = {i_prime_ma:.2f}$ mA, $P' = {p_prime_mw:.2f}$ mW."
        )

    with plot_col:
        st.markdown("**Solar Cell $V$-$I$ & Power Curves (with Load Line & MPP Rectangle)**")
        fig_iv, (ax_iv, ax_pv) = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)

        # Top Plot: V-I Curve + Max Power Rectangle + Load Line
        ax_iv.plot(model["v_arr"], model["i_mag"] * 1e3, color="#1f77b4", linewidth=2.3, label="Solar Cell $I$-$V$ Curve")
        ax_iv.fill_between([0, vm_v], 0, im_ma, color="#2ca02c", alpha=0.14, label=f"Max Power Rectangle ($I_m V_m = {pmax_mw:.2f}$ mW)")
        ax_iv.plot([0, vm_v, vm_v], [im_ma, im_ma, 0], color="#2ca02c", linestyle="--", linewidth=1.2)

        # Plot Load Line I = V / R if R > 0 and finite
        if 0 < r_load_ohm < np.inf:
            v_ll = np.linspace(0, voc_v, 100)
            i_ll_ma = (v_ll / r_load_ohm) * 1e3
            valid_ll = i_ll_ma <= isc_ma * 1.15
            ax_iv.plot(v_ll[valid_ll], i_ll_ma[valid_ll], color="#ff7f0e", linestyle=":", linewidth=1.6, label=f"Load Line ($R = {r_load_ohm:.0f}\\;\\Omega$)")

        ax_iv.scatter([v_prime], [i_prime_ma], color="#d62728", zorder=6, s=55, label=f"Operating Point P ($V'={v_prime:.2f}$V, $I'={i_prime_ma:.1f}$mA)")
        style_cro_axes(ax_iv, title="Solar Cell V-I Characteristic & Maximum Power Rectangle", xlabel="", ylabel="Current |I| (mA)")
        ax_iv.legend(loc="lower left", fontsize=7.5)

        # Bottom Plot: Output Power P = I * V vs Voltage V
        ax_pv.plot(model["v_arr"], model["p_out_w"] * 1e3, color="#d62728", linewidth=2.2, label="Delivered Power $P_{out} = I \\cdot V$ (mW)")
        ax_pv.scatter([vm_v], [pmax_mw], color="#2ca02c", zorder=6, s=55, label=f"Maximum Power $P_{{max}} = {pmax_mw:.2f}$ mW")
        ax_pv.scatter([v_prime], [p_prime_mw], color="#1f77b4", zorder=6, s=45, label=f"Load Power $P' = {p_prime_mw:.2f}$ mW")
        style_cro_axes(ax_pv, title="Output Power vs. Terminal Voltage", xlabel="Voltage V (V)", ylabel="Power Output (mW)")
        ax_pv.legend(loc="upper left", fontsize=7.5)

        plt.tight_layout()
        st.pyplot(fig_iv)
        plt.close(fig_iv)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Observation Table + 4th-Quadrant Plot + Lab Manual Theory
    # ---------------------------------------------------------------------
    tab_obs, tab_4q, tab_manual = st.tabs([
        "Observation Table (0 Ohm to Infinity)",
        "Fourth-Quadrant Characteristic (Eqs. 2 & 3)",
        "Aim, Theory, Procedure & Result"
    ])

    with tab_obs:
        st.markdown(
            f"**Tabular Column ($r = {dist_cm:.0f}$ cm, $A = {area_cm2:.1f}$ cm$^2$, $P_{{in}} = {model['p_in_mw']:.2f}$ mW):**"
        )
        manual_r_list = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, np.inf]
        rows = []
        for r_val in manual_r_list:
            v_r, i_r_a, p_r_mw = solve_for_load_resistance(r_val, model)
            i_r_ma = i_r_a * 1e3
            if r_val == 0:
                r_str = "0 (Short Circuit)"
                i_str = f"Isc = {i_r_ma:.2f}"
                v_str = f"{v_r:.3f}"
            elif np.isinf(r_val):
                r_str = "Infinity (Open Circuit)"
                i_str = f"{i_r_ma:.3f}"
                v_str = f"Voc = {v_r:.3f}"
            else:
                r_str = f"{int(r_val)}"
                i_str = f"{i_r_ma:.2f}"
                v_str = f"{v_r:.3f}"
            rows.append({
                "Load Resistance (Ohm)": r_str,
                "Current I in (mA)": i_str,
                "Voltage V in (volt)": v_str,
                "Power output = I x V (mW)": f"{p_r_mw:.2f}",
            })
        st.table(rows)

    with tab_4q:
        fig_4q, ax_4q = plt.subplots(figsize=(8.6, 3.8))
        ax_4q.plot(model["v_arr"], model["i_fourth_quad"] * 1e3, color="#1f77b4", linewidth=2.2, label="Eq. (2): $I = -I_{ph} + I_0(e^{eV/\\eta kT} - 1)$")
        if 0 < r_load_ohm < np.inf:
            v_ll = np.linspace(0, voc_v, 100)
            ax_4q.plot(v_ll, -(v_ll / r_load_ohm) * 1e3, color="#ff7f0e", linestyle="--", linewidth=1.6, label=f"Eq. (3) Load Line: $I = -V/R$ ($R = {r_load_ohm:.0f}\\;\\Omega$)")
        ax_4q.scatter([v_prime], [-i_prime_ma], color="#d62728", zorder=6, s=55, label=f"Operating Point P ($V'={v_prime:.3f}$V, $I'=-{i_prime_ma:.2f}$mA)")
        style_cro_axes(ax_4q, title="Fourth-Quadrant Solar Cell Characteristic & Load Line Intersection (Eqs. 2 & 3)", xlabel="Voltage V (V)", ylabel="Net Current I (mA)")
        ax_4q.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_4q)
        plt.close(fig_4q)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Results

            * **Components:** Solar Cell, variable resistors, milli-ammeter, voltmeter, incandescent bulb ({p_bulb_w:.0f} W).
            * **Working Equations:**
              1. **Dark Diode Current:** $I_{{\\text{{diode}}}} = I_0 \\left(e^{{\\frac{{eV}}{{\\eta k T}}}} - 1\\right)$
              2. **Net Illuminated Solar Cell Current:** $I = -I_{{ph}} + I_0 \\left(e^{{\\frac{{eV}}{{\\eta k T}}}} - 1\\right)$
              3. **Load Line Equation:** $I = -\\frac{{V}}{{R}}$
              4. **Fill Factor:** $FF = \\frac{{I_m V_m}}{{I_{{sc}} V_{{oc}}}} = \\frac{{{im_ma:.2f}\\text{{ mA}} \\times {vm_v:.3f}\\text{{ V}}}}{{{isc_ma:.2f}\\text{{ mA}} \\times {voc_v:.3f}\\text{{ V}}}} = {ff_val:.3f}$
              5. **Efficiency:** $\\text{{Efficiency}} = \\frac{{P_{{out}}}}{{P_{{in}}}} = \\frac{{{pmax_mw:.2f}\\text{{ mW}}}}{{{model['p_in_mw']:.2f}\\text{{ mW}}}} = {eff_pct:.2f}\\%$

            * **Result:**
              * **Fill Factor of the given solar cell** = **{ff_val:.3f}** (**{ff_val*100:.1f}%**)
              * **Efficiency of the given solar cell** = **{eff_pct:.2f} %**
            """
        )