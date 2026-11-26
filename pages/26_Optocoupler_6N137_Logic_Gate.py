import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 26: Optocoupler 6N137 as Logic Gate", layout="wide")
render_sidebar_header()

st.title("Experiment 26: High-Speed Optocoupler (6N137) as a Logic Gate")
st.markdown(
    "**Aim:** To understand the pin configuration and working of a high-speed optocoupler (**6N137**), "
    "configure it as an **Inverting Logic Gate** and a **Non-Inverting Buffer**, and verify the truth table "
    "using an output LED indicator."
)

# -------------------------------------------------------------------------
# Pure-Matplotlib Helper: Exact 6N137 Schematic (Matches Lab Manual Diagram)
# -------------------------------------------------------------------------
def draw_resistor_h(ax, x0, x1, y, label="", fontsize=8.0):
    xm = 0.5 * (x0 + x1)
    rw, rh = 0.72, 0.22
    ax.plot([x0, xm - rw / 2], [y, y], color="#0f172a", lw=1.5)
    ax.plot([xm + rw / 2, x1], [y, y], color="#0f172a", lw=1.5)
    ax.add_patch(
        patches.Rectangle(
            (xm - rw / 2, y - rh / 2), rw, rh,
            lw=1.4, ec="#0f172a", fc="#cbd5e1", zorder=4
        )
    )
    if label:
        ax.text(xm, y + 0.18, label, ha="center", va="bottom", fontsize=fontsize, color="#0f172a")


def draw_resistor_v(ax, x, y_top, y_bot, label="", fontsize=8.0):
    ym = 0.5 * (y_top + y_bot)
    rw, rh = 0.20, 0.42
    ax.plot([x, x], [y_top, ym + rh / 2], color="#0f172a", lw=1.5)
    ax.plot([x, x], [ym - rh / 2, y_bot], color="#0f172a", lw=1.5)
    ax.add_patch(
        patches.Rectangle(
            (x - rw / 2, ym - rh / 2), rw, rh,
            lw=1.4, ec="#0f172a", fc="#cbd5e1", zorder=4
        )
    )
    if label:
        ax.text(x + 0.16, ym, label, ha="left", va="center", fontsize=fontsize, color="#0f172a")


def draw_switch_h(ax, x0, x1, y, closed=True, label="S", fontsize=8.0):
    xm = 0.5 * (x0 + x1)
    sw = 0.55
    ax.plot([x0, xm - sw / 2], [y, y], color="#0f172a", lw=1.5)
    ax.plot([xm + sw / 2, x1], [y, y], color="#0f172a", lw=1.5)
    ax.scatter([xm - sw / 2, xm + sw / 2], [y, y], s=18, color="#0f172a", zorder=5)
    if closed:
        ax.plot([xm - sw / 2, xm + sw / 2], [y, y + 0.03], color="#0f172a", lw=1.8, zorder=5)
    else:
        ax.plot([xm - sw / 2, xm + sw / 2 - 0.05], [y, y + 0.28], color="#0f172a", lw=1.8, zorder=5)
    ax.text(xm, y + 0.24, label, ha="center", va="bottom", fontsize=fontsize, color="#0f172a")


def draw_6n137_schematic(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm, led_on):
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 1. Central 6N137 IC Package
    ic_l, ic_r = -1.45, 1.45
    ic_b, ic_t = -2.20, 2.20
    y1, y2, y3, y4 = 1.35, 0.45, -0.45, -1.35

    ax.add_patch(
        patches.Rectangle((ic_l, ic_b), ic_r - ic_l, ic_t - ic_b, lw=1.6, ec="#0f172a", fc="white", zorder=2)
    )

    # Left Pins: 1 (N/C), 2 (+), 3 (-), 4 (N/C)
    for pnum, py, ext_lbl, pol_lbl in [
        (1, y1, "N/C", ""),
        (2, y2, "", "+"),
        (3, y3, "", "-"),
        (4, y4, "N/C", ""),
    ]:
        ax.add_patch(patches.Rectangle((ic_l - 0.30, py - 0.15), 0.30, 0.30, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_l - 0.15, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        if ext_lbl:
            ax.text(ic_l - 0.38, py, ext_lbl, ha="right", va="center", fontsize=7.8, color="#0f172a")
        if pol_lbl:
            ax.text(ic_l - 0.38, py + 0.06, pol_lbl, ha="right", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

    ax.text(ic_l - 0.45, 0.0, "$V_F$", ha="right", va="center", fontsize=8.2, color="#0f172a")

    # Right Pins: 8 (VCC), 7 (VE), 6 (VO), 5 (GND)
    for pnum, py, ext_lbl in [
        (8, y1, "$V_{CC}$"),
        (7, y2, "$V_E$"),
        (6, y3, "$V_O$"),
        (5, y4, "GND"),
    ]:
        ax.add_patch(patches.Rectangle((ic_r, py - 0.15), 0.30, 0.30, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_r + 0.15, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        ax.text(ic_r + 0.38, py + 0.06, ext_lbl, ha="left", va="bottom", fontsize=7.8, color="#0f172a")

    # 2. Internal 6N137 Architecture (IR LED + Isolation + Photodiode + Amp + Gate + Open-Collector BJT)
    lx = -0.92
    ax.plot([ic_l, lx, lx], [y2, y2, 0.16], color="#0f172a", lw=1.3)
    ax.plot([lx, lx, ic_l], [-0.16, y3, y3], color="#0f172a", lw=1.3)
    # IR LED triangle & bar
    ax.add_patch(patches.Polygon([[lx - 0.14, 0.16], [lx + 0.14, 0.16], [lx, -0.12]], closed=True, lw=1.2, ec="#0f172a", fc="white", zorder=4))
    ax.plot([lx - 0.15, lx + 0.15], [-0.12, -0.12], color="#0f172a", lw=1.4, zorder=4)

    # Optical coupling arrows + dashed isolation line
    ax.annotate("", xy=(-0.44, 0.08), xytext=(-0.74, 0.00), arrowprops=dict(arrowstyle="->", lw=1.1, color="#0f172a"))
    ax.annotate("", xy=(-0.44, -0.06), xytext=(-0.74, -0.14), arrowprops=dict(arrowstyle="->", lw=1.1, color="#0f172a"))
    ax.plot([-0.58, -0.58], [ic_b + 0.22, ic_t - 0.22], linestyle="--", lw=1.1, color="#0f172a", zorder=3)

    # Detector side internal buses
    px, ax_amp, qx = -0.24, 0.16, 0.96
    ax.plot([px, ic_r], [y1, y1], color="#0f172a", lw=1.2)
    ax.plot([px, ic_r], [y4, y4], color="#0f172a", lw=1.2)
    ax.scatter([px, ax_amp, ax_amp, qx], [y1, y1, y4, y4], s=10, color="#0f172a", zorder=5)

    # Internal Photodiode (pointing up)
    ax.plot([px, px], [y1, 0.16], color="#0f172a", lw=1.2)
    ax.plot([px, px], [-0.14, -1.02], color="#0f172a", lw=1.2)
    ax.add_patch(patches.Polygon([[px - 0.12, -0.12], [px + 0.12, -0.12], [px, 0.14]], closed=True, lw=1.1, ec="#0f172a", fc="white", zorder=4))
    ax.plot([px - 0.13, px + 0.13], [0.14, 0.14], color="#0f172a", lw=1.3, zorder=4)

    # Internal Amplifier + Enable Gate + Open-Collector NPN
    ax.plot([ax_amp, ax_amp], [y1, y4], color="#0f172a", lw=1.1)
    ax.plot([px, 0.04], [-1.02, -1.02], color="#0f172a", lw=1.2)
    ax.add_patch(patches.Polygon([[0.04, -0.88], [0.04, -1.16], [0.30, -1.02]], closed=True, lw=1.1, ec="#0f172a", fc="white", zorder=4))
    ax.plot([0.30, 0.44], [-1.02, -1.02], color="#0f172a", lw=1.2)

    # Enable path from Pin 7 to AND gate
    ax.plot([ic_r, 0.36, 0.36, 0.44], [y2, y2, -0.92, -0.92], color="#0f172a", lw=1.1)
    ax.add_patch(patches.Rectangle((0.44, -1.12), 0.22, 0.24, lw=1.1, ec="#0f172a", fc="white", zorder=4))
    ax.plot([0.66, qx], [-1.00, -1.00], color="#0f172a", lw=1.2)

    # Output NPN Transistor
    ax.plot([qx, qx], [-1.18, -0.82], color="#0f172a", lw=1.8)
    ax.plot([qx, qx + 0.22, qx + 0.22, ic_r], [-0.90, -0.72, y3, y3], color="#0f172a", lw=1.2)
    ax.plot([qx, qx + 0.22, qx + 0.22], [-1.10, -1.26, y4], color="#0f172a", lw=1.2)

    # 3. External Input Side (Left)
    p2_x, p3_x = ic_l - 0.30, ic_l - 0.30
    bat_x = -4.95

    if config_mode.startswith("Inverting"):
        # 5V -> RIN -> Switch S -> Pin 2 (+); Pin 3 (-) -> 5V (-)
        draw_resistor_h(ax, bat_x, -3.35, y2, label=f"{rin_ohm:.0f}$\\Omega$")
        draw_switch_h(ax, -3.35, p2_x, y2, closed=vin_high, label="S")
        ax.plot([bat_x, p3_x], [y3, y3], color="#0f172a", lw=1.5)
    else:
        # Non-Inverting Buffer: 5V -> RIN -> Pin 2 (+); Pin 3 (-) -> Switch S -> 5V (-)
        draw_resistor_h(ax, bat_x, p2_x, y2, label=f"{rin_ohm:.0f}$\\Omega$")
        draw_switch_h(ax, bat_x, p3_x, y3, closed=(not vin_high), label="S")

    # Left 5V DC Supply (+ top, - bottom)
    ax.plot([bat_x, bat_x], [y2, 0.08], color="#0f172a", lw=1.5)
    ax.plot([bat_x, bat_x], [-0.08, y3], color="#0f172a", lw=1.5)
    ax.plot([bat_x - 0.22, bat_x + 0.22], [0.08, 0.08], color="#0f172a", lw=1.6)
    ax.plot([bat_x - 0.12, bat_x + 0.12], [-0.08, -0.08], color="#0f172a", lw=2.2)
    ax.text(bat_x - 0.30, 0.0, "5V", ha="right", va="center", fontsize=8.5, color="#0f172a")

    # 4. External Output Side (Right)
    p8_x = ic_r + 0.30
    bus_vcc_x = 4.65
    bat_r_x = 5.15
    ret_x = 6.15

    # Pin 8 (VCC) -> +5V Battery -> Right Return Bus -> Pin 5 (GND)
    ax.plot([p8_x, bus_vcc_x], [y1, y1], color="#0f172a", lw=1.5)
    ax.scatter([bus_vcc_x], [y1], s=18, color="#0f172a", zorder=5)

    # Right 5V Battery (+ left, - right)
    ax.plot([bus_vcc_x, bat_r_x - 0.08], [y1, y1], color="#0f172a", lw=1.5)
    ax.plot([bat_r_x + 0.08, ret_x], [y1, y1], color="#0f172a", lw=1.5)
    ax.plot([bat_r_x - 0.08, bat_r_x - 0.08], [y1 - 0.22, y1 + 0.22], color="#0f172a", lw=1.6)
    ax.plot([bat_r_x + 0.08, bat_r_x + 0.08], [y1 - 0.12, y1 + 0.12], color="#0f172a", lw=2.2)
    ax.text(bat_r_x, y1 + 0.28, "5V", ha="center", va="bottom", fontsize=8.5, color="#0f172a")

    ax.plot([ret_x, ret_x, p8_x], [y1, y4, y4], color="#0f172a", lw=1.5)

    # Ground symbol on bottom return bus
    gnd_x = 4.35
    ax.scatter([gnd_x], [y4], s=18, color="#0f172a", zorder=5)
    ax.plot([gnd_x, gnd_x], [y4, y4 - 0.38], color="#0f172a", lw=1.5)
    ax.plot([gnd_x - 0.26, gnd_x + 0.26], [y4 - 0.38, y4 - 0.38], color="#0f172a", lw=1.6)
    ax.plot([gnd_x - 0.17, gnd_x + 0.17], [y4 - 0.48, y4 - 0.48], color="#0f172a", lw=1.5)
    ax.plot([gnd_x - 0.08, gnd_x + 0.08], [y4 - 0.58, y4 - 0.58], color="#0f172a", lw=1.4)

    # Vertical +5V bus down to Pull-up resistor level
    y_pull = -0.32
    ax.plot([bus_vcc_x, bus_vcc_x], [y1, y_pull], color="#0f172a", lw=1.5)

    # Pin 7 (Enable VE) -> Switch S -> +5V bus (or GND if L)
    if enable_state == "H (+5V)":
        draw_switch_h(ax, p8_x, bus_vcc_x, y2, closed=True, label="S")
        ax.scatter([bus_vcc_x], [y2], s=18, color="#0f172a", zorder=5)
    elif enable_state == "L (GND)":
        ax.plot([p8_x, 2.35, 2.35], [y2, y2, y4], color="#0f172a", lw=1.5)
        ax.scatter([2.35], [y4], s=18, color="#0f172a", zorder=5)
        ax.text(2.42, 0.0, "$V_E=L$", ha="left", va="center", fontsize=7.6, color="#0f172a")
    else:
        draw_switch_h(ax, p8_x, bus_vcc_x, y2, closed=False, label="S (NC)")

    # Pin 6 (VO) -> Branch to 1 kOhm Pull-up & 220 Ohm + LED
    bx = 2.65
    ax.plot([p8_x, bx], [y3, y3], color="#0f172a", lw=1.5)
    ax.scatter([bx], [y3], s=18, color="#0f172a", zorder=5)

    # Pull-up resistor RPULL to +5V vertical bus
    ax.plot([bx, bx], [y3, y_pull], color="#0f172a", lw=1.5)
    rpull_lbl = f"{rpull_ohm/1000:.1f}k$\\Omega$" if rpull_ohm >= 1000 else f"{rpull_ohm:.0f}$\\Omega$"
    draw_resistor_h(ax, bx, bus_vcc_x, y_pull, label=rpull_lbl, fontsize=8.2)

    # 220 Ohm + Output LED down to GND bus
    y_led_top = -0.96
    draw_resistor_v(ax, bx, y3, y_led_top, label=f"{rled_ohm:.0f}$\\Omega$", fontsize=8.0)

    y_led_mid = -1.16
    ax.plot([bx, bx], [y_led_top, y_led_mid + 0.11], color="#0f172a", lw=1.5)
    ax.plot([bx, bx], [y_led_mid - 0.11, y4], color="#0f172a", lw=1.5)
    led_fc = "#ef4444" if led_on else "white"
    ax.add_patch(
        patches.Polygon(
            [[bx - 0.13, y_led_mid + 0.11], [bx + 0.13, y_led_mid + 0.11], [bx, y_led_mid - 0.11]],
            closed=True, lw=1.3, ec="#0f172a", fc=led_fc, zorder=4
        )
    )
    ax.plot([bx - 0.14, bx + 0.14], [y_led_mid - 0.11, y_led_mid - 0.11], color="#0f172a", lw=1.5, zorder=4)
    ax.annotate("", xy=(bx + 0.32, y_led_mid - 0.04), xytext=(bx + 0.16, y_led_mid + 0.06), arrowprops=dict(arrowstyle="->", lw=1.0, color="#0f172a"))
    ax.annotate("", xy=(bx + 0.32, y_led_mid - 0.16), xytext=(bx + 0.16, y_led_mid - 0.06), arrowprops=dict(arrowstyle="->", lw=1.0, color="#0f172a"))
    ax.text(bx + 0.40, y_led_mid - 0.05, "LED (ON)" if led_on else "LED", ha="left", va="center", fontsize=8.0, color="#0f172a")
    ax.scatter([bx], [y4], s=18, color="#0f172a", zorder=5)

    ax.set_xlim(-5.55, 6.55)
    ax.set_ylim(-2.45, 2.45)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


# -------------------------------------------------------------------------
# Physics & Logic Model for 6N137 High-Speed Optocoupler
# -------------------------------------------------------------------------
def evaluate_6n137(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm):
    vcc = 5.0
    vf_ir = 1.50
    if_thresh_ma = 5.0
    vol_sat = 0.25
    vf_ind = 2.00

    ir_driven = vin_high if config_mode.startswith("Inverting") else (not vin_high)
    if_ma = ((vcc - vf_ir) / rin_ohm) * 1e3 if ir_driven else 0.0
    ir_led_on = ir_driven and (if_ma >= if_thresh_ma)

    if enable_state == "L (GND)":
        out_high = True
    else:
        out_high = not ir_led_on

    if out_high:
        i_led_ma = max((vcc - vf_ind) / (rpull_ohm + rled_ohm), 0.0) * 1e3
        vout_v = vf_ind + (i_led_ma * 1e-3) * rled_ohm
        ind_led_on = i_led_ma > 1.0
    else:
        vout_v = vol_sat
        i_led_ma = 0.0
        ind_led_on = False

    return {
        "ir_led_on": ir_led_on,
        "if_ma": if_ma,
        "out_high": out_high,
        "vout_v": vout_v,
        "i_led_ma": i_led_ma,
        "ind_led_on": ind_led_on,
    }


# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Truth Table)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Gate Configuration")
    config_mode = st.radio(
        "Select 6N137 Logic Configuration",
        [
            "Inverting Logic Gate (Standard Anode Drive)",
            "Non-Inverting Buffer (Cathode Drive)",
        ],
        index=0,
    )

    st.subheader("2. Digital Input & Enable Switches")
    vin_choice = st.radio("Digital Input $V_{\\text{IN}}$ (Switch S)", ["LOW (0 V)", "HIGH (+5 V)"], horizontal=True, index=1)
    vin_high = vin_choice.startswith("HIGH")

    enable_state = st.selectbox(
        "Enable Input $V_E$ (Pin 7 Switch S)",
        ["H (+5V)", "L (GND)", "NC (Open)"],
        index=0,
        help="Pin 7 (Enable): When H (+5V) or NC (Open), output is enabled. When L (GND), output is forced HIGH."
    )

    st.subheader("3. Resistor Values")
    rin_ohm = st.slider(
        "Input Current-Limiting Resistor $R_{\\text{IN}}$ (Ohm)",
        min_value=220,
        max_value=1000,
        value=470,
        step=10,
        help="RIN = (5V - 1.5V) / 10 mA = 350 Ohm; 470 Ohm is standard (IF = 7.45 mA)."
    )
    rpull_ohm = st.slider("Output Pull-Up Resistor $R_{\\text{PULL}}$ (Ohm)", 470, 4700, 1000, 100)
    rled_ohm = st.slider("Indicator LED Resistor $R_{\\text{LED}}$ (Ohm)", 150, 470, 220, 10)

res = evaluate_6n137(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm)

with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internal IR LED", "ON" if res["ir_led_on"] else "OFF")
    m2.metric("Input Forward Current $I_F$", f"{res['if_ma']:.2f} mA")
    m3.metric("Output Logic (Pin 6)", "HIGH (H)" if res["out_high"] else "LOW (L)")
    m4.metric("Output Voltage $V_O$", f"{res['vout_v']:.2f} V")

    if res["ind_led_on"]:
        st.success(
            f"**Output HIGH ({res['vout_v']:.2f} V) -- Indicator LED ON ($I_{{\\text{{LED}}}} = {res['i_led_ma']:.2f}$ mA):** "
            f"Internal IR LED is **{'ON' if res['ir_led_on'] else 'OFF'}** ($I_F = {res['if_ma']:.2f}$ mA), Enable (Pin 7) = **{enable_state}**."
        )
    else:
        st.info(
            f"**Output LOW ({res['vout_v']:.2f} V) -- Indicator LED OFF ($I_{{\\text{{LED}}}} = 0.00$ mA):** "
            f"Internal IR LED is **ON** ($I_F = {res['if_ma']:.2f}$ mA) and Enable (Pin 7) = **{enable_state}**, pulling Pin 6 LOW."
        )

    schem_col, tt_col = st.columns([1.3, 1.2], gap="medium")

    with schem_col:
        st.markdown(f"**6N137 Schematic ({config_mode.split(' (')[0]})**")
        sch_buf = draw_6n137_schematic(
            config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm, res["ind_led_on"]
        )
        st.image(sch_buf, use_container_width=True)
        st.caption(
            "Input 5V loop (Pins 2 & 3) is optically isolated from the output 5V circuit (Pins 5–8)."
        )

    with tt_col:
        st.markdown("**Truth Table Verification (6N137)**")
        truth_rows = [
            {"LED": "ON",  "Enable": "H",  "Out": "L", "Indicator LED": "OFF"},
            {"LED": "OFF", "Enable": "H",  "Out": "H", "Indicator LED": "ON"},
            {"LED": "ON",  "Enable": "L",  "Out": "H", "Indicator LED": "ON"},
            {"LED": "OFF", "Enable": "L",  "Out": "H", "Indicator LED": "ON"},
            {"LED": "ON",  "Enable": "NC", "Out": "L", "Indicator LED": "OFF"},
            {"LED": "OFF", "Enable": "NC", "Out": "H", "Indicator LED": "ON"},
        ]

        cur_led = "ON" if res["ir_led_on"] else "OFF"
        cur_en = enable_state.split(" ")[0]
        for row in truth_rows:
            row["Status"] = "<-- ACTIVE" if (row["LED"] == cur_led and row["Enable"] == cur_en) else ""

        st.table(truth_rows)

        st.markdown(
            """
            * **Inverting Gate:** Switch $S$ closed ($V_{\\text{IN}} = 5\\text{ V}$) $\\Rightarrow$ IR LED ON $\\Rightarrow V_O = \\text{LOW}$ (LED OFF).
            * **Non-Inverting Buffer:** Switch $S$ open ($V_{\\text{IN}} = 5\\text{ V}$) $\\Rightarrow$ IR LED OFF $\\Rightarrow V_O = \\text{HIGH}$ (LED ON).
            """
        )

    tab_curve, tab_manual = st.tabs([
        "Input Resistor Design Curve (IF vs. RIN)",
        "Aim, Pin Configuration, Theory & Procedure"
    ])

    with tab_curve:
        rin_sweep = np.linspace(150, 1000, 300)
        if_sweep = ((5.0 - 1.5) / rin_sweep) * 1e3

        fig_if, ax_if = plt.subplots(figsize=(8.6, 3.6))
        ax_if.plot(rin_sweep, if_sweep, color="#1f77b4", linewidth=2.2, label="$I_F = (V_{IN} - V_F) / R_{IN}$ ($V_{IN}=5$ V, $V_F=1.5$ V)")
        ax_if.axhspan(5.0, 15.0, color="#2ca02c", alpha=0.14, label="Recommended 6N137 Forward Current Range (5–15 mA)")
        ax_if.scatter([rin_ohm], [(5.0 - 1.5) / rin_ohm * 1e3], color="#d62728", zorder=6, s=55, label=f"Selected $R_{{IN}} = {rin_ohm}\\;\\Omega$ ($I_F = {(5.0-1.5)/rin_ohm*1e3:.2f}$ mA)")
        style_cro_axes(ax_if, title="6N137 Input LED Forward Current IF vs. Limiting Resistor RIN", xlabel="Input Resistor RIN (Ohm)", ylabel="Forward Current IF (mA)")
        ax_if.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_if, clear_figure=True)
        plt.close(fig_if)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Pin Configuration

            * **Pin Configuration of 6N137 (8-Pin DIP):**
              * **Pins 1 & 4 (N/C):** No Connection.
              * **Pin 2 (Anode +) & Pin 3 (Cathode -):** Input infrared LED ($I_F = 5\\text{{--}}15\\text{{ mA}}$, $V_F \\approx 1.5\\text{{ V}}$).
              * **Pin 5 (GND):** Output-side ground.
              * **Pin 6 (Output $V_O$):** Open-collector logic output (requires pull-up resistor $R_{{PULL}} = 1\\text{{ k}}\\Omega$ to $V_{{CC}}$).
              * **Pin 7 (Enable $V_E$):** Active-low enable input (when left open/`NC` or connected `HIGH`, output is enabled; when connected `LOW`, output is forced high-impedance / `HIGH`).
              * **Pin 8 ($V_{{CC}}$):** Output-side supply ($+5\\text{{ V}}$ typical), decoupled to Pin 5 with a $0.1\\;\\mu\\text{{F}}$ bypass capacitor.

            * **Input Resistor Design Calculation:**
              For $V_{{IN}} = 5\\text{{ V}}$, $V_F \\approx 1.5\\text{{ V}}$, and desired $I_F = 10\\text{{ mA}}$:
              $$R_{{IN}} = \\frac{{5\\text{{ V}} - 1.5\\text{{ V}}}}{{0.01\\text{{ A}}}} = 350\\;\\Omega$$
              A standard **$470\\;\\Omega$** resistor gives $I_F = \\frac{{3.5\\text{{ V}}}}{{470\\;\\Omega}} = 7.45\\text{{ mA}}$ (within the $5\\text{{--}}15\\text{{ mA}}$ operating range).
            """
        )