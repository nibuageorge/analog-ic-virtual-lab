import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 22: Constant Current LED Driver", layout="wide")
render_sidebar_header()

st.title("Experiment 22: Constant Current LED Driver using Op-Amp & Transistor")
st.markdown(
    "**Aim:** To design and construct a closed-loop **Constant Current LED Driver** using an operational amplifier "
    "(**LM358**) and NPN transistor (**BC547 / BC548**), and study the stability of LED current ($I_{\\text{LED}}$) "
    "with varying supply voltage ($V_+$) and LED load conditions."
)

# -------------------------------------------------------------------------
# Helper: Draw Constant Current LED Driver Schematic
# -------------------------------------------------------------------------
def draw_led_driver_schematic(vin_v, rsense_ohm, vplus_v, led_color_label, n_leds):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.0)

        # Op-amp (LM358): Non-inverting (+) at bottom, Inverting (-) at top
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label("LM358", loc="center", fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.32).label(f"+{vplus_v:.1f}V", loc="top", fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.32).label("GND", loc="bottom", fontsize=7.5))

        # Control Voltage Vin at non-inverting (+) input
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.95))
        src_x = d.here[0]
        d.add(elm.SourceV().down().at((src_x, noninv_y)).length(1.35).label(f"$V_{{in}}$\n{vin_v:.2f} V", loc="left", fontsize=8.2))
        d.add(elm.Ground())

        # Op-amp output -> Base resistor -> NPN Transistor (BC547)
        d.add(elm.Resistor().right().at(op.out).length(1.15).label("$R_B$\n1 k$\\Omega$", loc="top", fontsize=7.8))
        q1 = d.add(elm.BjtNpn(circle=True).anchor("base").label("BC547", loc="right", fontsize=8.0))

        # Collector branch: V+ -> mA meter -> LED(s) -> Collector
        cx, cy = q1.collector
        d.add(elm.LED().up().at((cx, cy)).length(1.15).reverse().label(f"{n_leds}x {led_color_label}\nLED", loc="right", fontsize=7.8))
        d.add(elm.MeterA().up().length(1.05).label("mA", loc="right", fontsize=8.0))
        d.add(elm.Line().up().length(0.3).label(f"$V_+$ = {vplus_v:.1f} V", loc="top", fontsize=8.5))

        # Emitter branch: Emitter -> Sense Node -> Rsense -> Ground
        ex, ey = q1.emitter
        d.add(elm.Line().down().at((ex, ey)).length(0.35))
        sense_dot = d.add(elm.Dot())
        sx, sy = sense_dot.center

        d.add(elm.Resistor().down().at((sx, sy)).length(1.25).label(f"$R_{{sense}}$\n{rsense_ohm:.0f} $\\Omega$", loc="right", fontsize=8.2))
        d.add(elm.Ground())

        # Feedback line from Sense Node to Inverting (-) input
        inv_x, inv_y = op.in1
        top_fb_y = inv_y + 1.15
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.55))
        fb_left_x = d.here[0]
        d.add(elm.Line().up().at((fb_left_x, inv_y)).toy(top_fb_y))
        d.add(elm.Line().right().tox(sx + 0.95).label("Feedback ($V_{sense} = V_{in}$)", loc="top", fontsize=7.6))
        fb_right_x = d.here[0]
        d.add(elm.Line().down().toy(sy))
        d.add(elm.Line().left().tox(sx))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Physics Model for Closed-Loop LED Current Driver
# -------------------------------------------------------------------------
LED_TYPES = {
    "Red (650 nm, Vf ~ 1.9 V)": 1.90,
    "Orange (610 nm, Vf ~ 2.0 V)": 2.00,
    "Yellow (590 nm, Vf ~ 2.1 V)": 2.10,
    "Green (530 nm, Vf ~ 2.4 V)": 2.40,
    "Blue (470 nm, Vf ~ 3.1 V)": 3.10,
    "White (Vf ~ 3.2 V)": 3.20,
}

def compute_led_driver_state(vplus_v, vin_v, rsense_ohm, vf_single_v, n_leds, beta_npn=220.0):
    vce_sat = 0.25
    v_led_total = n_leds * vf_single_v

    # Target closed-loop current: I_LED = Vin / Rsense
    i_target_a = vin_v / rsense_ohm
    v_sense_target = vin_v

    # Minimum required supply voltage: V+ > V_LED + V_CE(sat) + V_sense
    vplus_min = v_led_total + vce_sat + v_sense_target

    # LM358 maximum output swing is ~ (V+ - 1.5 V); base-emitter drop Vbe ~ 0.7 V
    v_op_max = max(vplus_v - 1.5, 0.0)
    v_sense_op_limit = max(v_op_max - 0.7, 0.0)

    # Collector-emitter compliance limit: V_sense_max = V+ - V_LED - V_CE(sat)
    v_sense_ce_limit = max(vplus_v - v_led_total - vce_sat, 0.0)

    v_sense_actual = min(v_sense_target, v_sense_ce_limit, v_sense_op_limit)
    i_led_actual_a = v_sense_actual / rsense_ohm

    # Account for small base current fraction (Ic = beta/(beta+1) * Ie)
    i_led_actual_a *= (beta_npn / (beta_npn + 1.0))

    in_regulation = (vplus_v >= vplus_min) and (v_sense_actual >= 0.98 * v_sense_target)
    p_rsense_mw = (i_led_actual_a ** 2) * rsense_ohm * 1e3
    vce_actual = max(vplus_v - v_led_total - v_sense_actual, vce_sat)
    p_bjt_mw = vce_actual * i_led_actual_a * 1e3

    return {
        "i_target_ma": i_target_a * 1e3,
        "i_led_ma": i_led_actual_a * 1e3,
        "v_sense": v_sense_actual,
        "v_led_total": v_led_total,
        "vplus_min": vplus_min,
        "in_regulation": in_regulation,
        "p_rsense_mw": p_rsense_mw,
        "p_bjt_mw": p_bjt_mw,
        "vce_actual": vce_actual,
    }

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Current Setpoint & Sense Resistor")
    vin_v = st.slider("Control Voltage $V_{in}$ (V)", 0.20, 3.00, 1.00, 0.05)
    rsense_ohm = st.slider(
        "Sense Resistor $R_{sense}$ (Ohm)",
        min_value=10.0,
        max_value=220.0,
        value=68.0,
        step=2.0,
        help="Standard 68 Ohm resistor with Vin = 1.0 V yields I_LED = 1.0 / 68 = 14.71 mA (~15 mA)."
    )

    st.subheader("2. Supply Voltage & LED Load")
    vplus_v = st.slider("LED Supply Voltage $V_+$ (V)", 1.5, 15.0, 5.0, 0.25)
    led_choice = st.selectbox("LED Type / Color", list(LED_TYPES.keys()), index=1)
    vf_single = LED_TYPES[led_choice]
    n_leds = st.slider("Number of Series LEDs (Load Variation)", 1, 4, 1, 1)

# -------------------------------------------------------------------------
# Compute Operating State
# -------------------------------------------------------------------------
state = compute_led_driver_state(vplus_v, vin_v, rsense_ohm, vf_single, n_leds)
led_short_name = led_choice.split(" (")[0]

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Measured $I_{\\text{LED}}$", f"{state['i_led_ma']:.2f} mA")
    m2.metric("Designed $I = V_{in}/R_{sense}$", f"{state['i_target_ma']:.2f} mA")
    m3.metric("Sense Voltage $V_{sense}$", f"{state['v_sense']:.3f} V")
    m4.metric("Min Supply $V_{+(min)}$", f"{state['vplus_min']:.2f} V")

    if state["in_regulation"]:
        st.success(
            f"**Constant Current Regulation Active:** $V_+ = {vplus_v:.2f}$ V $> V_{{+(min)}} = {state['vplus_min']:.2f}$ V. "
            f"Op-amp locks $V_{{sense}} = V_{{in}} = {state['v_sense']:.2f}$ V, maintaining **$I_{{\\text{{LED}}}} = {state['i_led_ma']:.2f}$ mA** "
            f"($P_{{Rsense}} = {state['p_rsense_mw']:.1f}$ mW, $P_{{BC547}} = {state['p_bjt_mw']:.1f}$ mW)."
        )
    else:
        st.warning(
            f"**Dropout / Below Compliance Voltage ($V_+ = {vplus_v:.2f}$ V < $V_{{+(min)}} = {state['vplus_min']:.2f}$ V):** "
            f"Transistor BC547 enters saturation ($V_{{CE(sat)}} \\approx 0.25$ V) and $I_{{\\text{{LED}}}}$ drops to **{state['i_led_ma']:.2f} mA**. "
            f"Increase $V_+$ above **{state['vplus_min']:.2f} V** for constant-current regulation."
        )

    schem_col, plot_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (LM358 + BC547)**")
        fig_sch = draw_led_driver_schematic(vin_v, rsense_ohm, vplus_v, led_short_name, n_leds)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Governing Equation:** $I_{{\\text{{LED}}}} = \\frac{{V_{{in}}}}{{R_{{sense}}}} = "
            f"\\frac{{{vin_v:.2f}\\text{{ V}}}}{{{rsense_ohm:.0f}\\;\\Omega}} = {state['i_target_ma']:.2f}\\text{{ mA}}$."
        )

    with plot_col:
        st.markdown("**Line & Load Regulation Curves ($I_{\\text{LED}}$ vs. $V_+$ & $V_{in}$)**")
        fig_reg, (ax_line, ax_ctrl) = plt.subplots(2, 1, figsize=(7.6, 5.4))

        # Top Plot: Line Regulation (I_LED vs Supply Voltage V+) across LED colors
        v_sup_sweep = np.linspace(1.0, 15.0, 300)
        for label_c, vf_c, col_c in [
            ("Red (1.9 V)", 1.90, "#d62728"),
            ("Green (2.4 V)", 2.40, "#2ca02c"),
            ("Blue (3.1 V)", 3.10, "#1f77b4"),
        ]:
            i_curve = [compute_led_driver_state(vs, vin_v, rsense_ohm, vf_c, n_leds)["i_led_ma"] for vs in v_sup_sweep]
            ax_line.plot(v_sup_sweep, i_curve, color=col_c, linewidth=2.0, label=f"{n_leds}x {label_c}")

        ax_line.axvline(state["vplus_min"], color="#64748b", linestyle="--", linewidth=1.1, label=f"$V_{{+(min)}} = {state['vplus_min']:.2f}$ V")
        ax_line.scatter([vplus_v], [state["i_led_ma"]], color="#000000", zorder=6, s=50, label=f"Operating Point ({vplus_v:.1f} V, {state['i_led_ma']:.2f} mA)")
        style_cro_axes(ax_line, title="Supply Stability: LED Current vs. Supply Voltage V+", xlabel="Supply Voltage V+ (V)", ylabel="I_LED (mA)")
        ax_line.legend(loc="lower right", fontsize=7.3)

        # Bottom Plot: Linear Current Control (I_LED vs Control Voltage Vin)
        vin_sweep = np.linspace(0.0, 3.0, 200)
        i_vin_curve = [compute_led_driver_state(vplus_v, vi, rsense_ohm, vf_single, n_leds)["i_led_ma"] for vi in vin_sweep]
        ax_ctrl.plot(vin_sweep, i_vin_curve, color="#9467bd", linewidth=2.2, label=f"$I_{{LED}} = V_{{in}} / R_{{sense}}$ ($R_{{sense}} = {rsense_ohm:.0f}\\;\\Omega$)")
        ax_ctrl.scatter([vin_v], [state["i_led_ma"]], color="#d62728", zorder=6, s=50, label=f"Current Setpoint ($V_{{in}} = {vin_v:.2f}$ V)")
        style_cro_axes(ax_ctrl, title="Linear Current Programming: I_LED vs. Control Voltage Vin", xlabel="Control Voltage Vin (V)", ylabel="I_LED (mA)")
        ax_ctrl.legend(loc="upper left", fontsize=7.5)

        plt.tight_layout()
        st.pyplot(fig_reg)
        plt.close(fig_reg)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Observation Tables + Theory & Design Write-Up
    # ---------------------------------------------------------------------
    tab_obs1, tab_obs2, tab_manual = st.tabs([
        "Observation Table 1: Supply Voltage Variation (Line Regulation)",
        "Observation Table 2: LED Load & Control Voltage Variation",
        "Aim, Theory, Design & Working"
    ])

    with tab_obs1:
        st.markdown(
            f"**Line Regulation Table ($V_{{in}} = {vin_v:.2f}$ V, $R_{{sense}} = {rsense_ohm:.0f}\\;\\Omega$, "
            f"Designed $I_{{\\text{{LED}}}} = {state['i_target_ma']:.2f}$ mA, Load = {n_leds}x {led_short_name} LED):**"
        )
        vplus_steps = [2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
        rows_vplus = []
        for vp in vplus_steps:
            st_vp = compute_led_driver_state(vp, vin_v, rsense_ohm, vf_single, n_leds)
            rows_vplus.append({
                "Supply Voltage V+ (V)": f"{vp:.1f}",
                "Vsense (V)": f"{st_vp['v_sense']:.3f}",
                "Designed I_LED (mA)": f"{st_vp['i_target_ma']:.2f}",
                "Measured I_LED (mA)": f"{st_vp['i_led_ma']:.2f}",
                "Status": "Regulated (Constant)" if st_vp["in_regulation"] else "Below V+(min) Dropout",
            })
        st.table(rows_vplus)

    with tab_obs2:
        st.markdown(
            f"**Load Variation Across Different LED Colors at $V_+ = {vplus_v:.1f}$ V, $V_{{in}} = {vin_v:.2f}$ V, $R_{{sense}} = {rsense_ohm:.0f}\\;\\Omega$:**"
        )
        rows_load = []
        for name_l, vf_l in LED_TYPES.items():
            st_l = compute_led_driver_state(vplus_v, vin_v, rsense_ohm, vf_l, 1)
            rows_load.append({
                "LED Type": name_l,
                "Forward Drop Vf (V)": f"{vf_l:.2f}",
                "Min Required V+ (V)": f"{st_l['vplus_min']:.2f}",
                "Measured I_LED (mA)": f"{st_l['i_led_ma']:.2f}",
                "Status": "Constant Current Maintained" if st_l["in_regulation"] else "Increase V+",
            })
        st.table(rows_load)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Design Verification

            * **Components:** Op-amp **LM358**, NPN Transistor **BC547 / BC548**, Sense Resistor $R_{{sense}} = 68\\;\\Omega$ ($0.25\\text{{ W}}$), Base Resistor $R_B = 1\\text{{ k}}\\Omega$, Standard $5\\text{{ mm}}$ LEDs, Regulated DC Power Supplies ($V_+$ and $V_{{in}}$), Digital Milliammeter.

            * **Governing Equations & Design:**
              1. **Closed-Loop Current Equation:**
                 $$V_{{sense}} = V_{{in}} \\implies I_{{\\text{{LED}}}} = \\frac{{V_{{sense}}}}{{R_{{sense}}}} = \\frac{{V_{{in}}}}{{R_{{sense}}}}$$
              2. **Component Sizing (for $I_{{\\text{{LED}}}} = 15\\text{{ mA}}$ at $V_{{in}} = 1\\text{{ V}}$):**
                 $$R_{{sense}} = \\frac{{1\\text{{ V}}}}{{0.015\\text{{ A}}}} = 66.7\\;\\Omega \\;(\\text{{nearest standard value: }} 68\\;\\Omega)$$
                 $$P_{{Rsense}} = I^2 R_{{sense}} = (0.015)^2 \\times 68 = 15.3\\text{{ mW}} \\ll 250\\text{{ mW}} \\;(0.25\\text{{ W resistor}})$$
              3. **Supply Voltage Compliance Condition:**
                 $$V_+ > V_{{\\text{{LED}}}} + V_{{CE(\\text{{sat}})}} + V_{{sense}}$$
                 For $V_{{\\text{{LED}}}} = 2.0\\text{{ V}}$, $V_{{sense}} = 1.0\\text{{ V}}$, and $V_{{CE(\\text{{sat}})}} \\approx 0.25\\text{{ V}}$, $V_+ > 3.25\\text{{ V}}$ (so $V_+ = 5\\text{{ V}}$ ensures stable constant-current operation).

            * **Result:**
              * Designed LED current $= {state['i_target_ma']:.2f}\\text{{ mA}}$; Measured LED current $= {state['i_led_ma']:.2f}\\text{{ mA}}$ at $V_+ = {vplus_v:.1f}\\text{{ V}}$.
              * Once $V_+ > V_{{+(min)}} = {state['vplus_min']:.2f}\\text{{ V}}$, the LED current remains constant independent of variations in supply voltage $V_+$ and LED forward voltage $V_F$.
            """
        )