import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 27: Optocoupler PC817 - CTR & Characteristics", layout="wide")
render_sidebar_header()

st.title("Experiment 27: Optocoupler (PC817) -- Current Transfer Ratio (CTR) & Characteristics")
st.markdown(
    "**Aim:** To understand the construction and working principle of an optocoupler (**PC817**), "
    "determine its **Current Transfer Ratio ($\text{CTR} = \frac{I_C}{I_F} \times 100\%$)**, and plot the "
    "**Input ($V_F$ vs. $I_F$)**, **Output ($V_{CE}$ vs. $I_C$)**, and **Transfer ($I_F$ vs. $I_C$)** characteristic curves."
)

# -------------------------------------------------------------------------
# Pure-Matplotlib Helper: PC817 Optocoupler Characteristic Test Circuit
# -------------------------------------------------------------------------
def draw_resistor_h(ax, x0, x1, y, label="", fontsize=8.0):
    xm = 0.5 * (x0 + x1)
    rw, rh = 0.68, 0.20
    ax.plot([x0, xm - rw / 2], [y, y], color="#0f172a", lw=1.5)
    ax.plot([xm + rw / 2, x1], [y, y], color="#0f172a", lw=1.5)
    ax.add_patch(
        patches.Rectangle(
            (xm - rw / 2, y - rh / 2), rw, rh,
            lw=1.4, ec="#0f172a", fc="#cbd5e1", zorder=4
        )
    )
    if label:
        ax.text(xm, y + 0.17, label, ha="center", va="bottom", fontsize=fontsize, color="#0f172a")


def draw_resistor_v(ax, x, y_top, y_bot, label="", fontsize=8.0):
    ym = 0.5 * (y_top + y_bot)
    rw, rh = 0.20, 0.56
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


def draw_meter(ax, x, y, text, fontsize=7.8):
    ax.add_patch(patches.Circle((x, y), 0.24, lw=1.4, ec="#0f172a", fc="white", zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#0f172a", zorder=6)


def draw_pc817_schematic(vin_v, vcc_v, rin_ohm, rl_ohm, vf_v, if_ma, vce_v, ic_ma):
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 1. Central 4-Pin PC817 Optocoupler Package
    # Pins: 1 (Anode +), 2 (Cathode -) on left; 4 (Collector C), 3 (Emitter E) on right
    ic_l, ic_r = -1.25, 1.25
    ic_b, ic_t = -1.55, 1.55
    y_top, y_bot = 0.85, -0.85

    ax.add_patch(
        patches.Rectangle((ic_l, ic_b), ic_r - ic_l, ic_t - ic_b, lw=1.6, ec="#0f172a", fc="white", zorder=2)
    )
    ax.text(0.0, ic_t + 0.14, "PC817 Optocoupler", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

    # Pin boxes and labels
    for pnum, py, plbl in [(1, y_top, "+"), (2, y_bot, "-")]:
        ax.add_patch(patches.Rectangle((ic_l - 0.28, py - 0.14), 0.28, 0.28, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_l - 0.14, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        ax.text(ic_l - 0.35, py + 0.08, plbl, ha="right", va="bottom", fontsize=8.2, fontweight="bold", color="#0f172a")

    for pnum, py, plbl in [(4, y_top, "C"), (3, y_bot, "E")]:
        ax.add_patch(patches.Rectangle((ic_r, py - 0.14), 0.28, 0.28, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_r + 0.14, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        ax.text(ic_r + 0.35, py + 0.08, plbl, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color="#0f172a")

    # Internal GaAs IR LED (Left side inside package)
    lx = -0.62
    ax.plot([ic_l, lx, lx], [y_top, y_top, 0.14], color="#0f172a", lw=1.4)
    ax.plot([lx, lx, ic_l], [-0.14, y_bot, y_bot], color="#0f172a", lw=1.4)
    ax.add_patch(patches.Polygon([[lx - 0.15, 0.14], [lx + 0.15, 0.14], [lx, -0.14]], closed=True, lw=1.3, ec="#0f172a", fc="white", zorder=4))
    ax.plot([lx - 0.16, lx + 0.16], [-0.14, -0.14], color="#0f172a", lw=1.5, zorder=4)

    # Optical Coupling Arrows + Dashed Isolation Barrier
    ax.annotate("", xy=(0.24, 0.12), xytext=(-0.34, 0.02), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0f172a"))
    ax.annotate("", xy=(0.24, -0.06), xytext=(-0.34, -0.16), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0f172a"))
    ax.plot([0.0, 0.0], [ic_b + 0.20, ic_t - 0.20], linestyle="--", lw=1.1, color="#64748b", zorder=3)

    # Internal Silicon NPN Phototransistor (Right side inside package)
    qx = 0.45
    ax.plot([qx, qx], [-0.36, 0.36], color="#0f172a", lw=2.0)  # Base bar
    ax.plot([qx, qx + 0.35, ic_r], [0.16, 0.45, y_top], color="#0f172a", lw=1.4)  # Collector
    ax.plot([qx, qx + 0.35, ic_r], [-0.16, -0.45, y_bot], color="#0f172a", lw=1.4)  # Emitter
    ax.annotate("", xy=(qx + 0.35, -0.45), xytext=(qx + 0.12, -0.26), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a"))

    # 2. Input Side Circuit (Left: VIN -> RIN -> Milliammeter mA -> Pin 1; Voltmeter VF across Pins 1-2; Pin 2 -> GND1)
    p1_x = ic_l - 0.28
    vf_x = -2.15
    ma_in_x = -3.05
    rin_l = -4.35
    vin_x = -5.05

    ax.plot([vin_x, rin_l], [y_top, y_top], color="#0f172a", lw=1.5)
    draw_resistor_h(ax, rin_l, ma_in_x - 0.24, y_top, label=f"$R_{{IN}}$ ({rin_ohm:.0f}$\\Omega$)", fontsize=7.8)
    ax.plot([ma_in_x - 0.24, p1_x], [y_top, y_top], color="#0f172a", lw=1.5)
    draw_meter(ax, ma_in_x, y_top, "mA")
    ax.text(ma_in_x, y_top + 0.32, f"$I_F$={if_ma:.1f}mA", ha="center", va="bottom", fontsize=7.4, color="#1d4ed8")

    # Voltmeter VF across Pins 1 and 2
    ax.plot([vf_x, vf_x], [y_top, y_bot], color="#0f172a", lw=1.3)
    ax.scatter([vf_x, vf_x], [y_top, y_bot], s=16, color="#0f172a", zorder=5)
    draw_meter(ax, vf_x, 0.0, "V")
    ax.text(vf_x - 0.30, 0.0, f"$V_F$\n{vf_v:.2f}V", ha="right", va="center", fontsize=7.4, color="#1d4ed8")

    # Input Variable DC Supply VIN
    ax.plot([vin_x, vin_x], [y_top, 0.10], color="#0f172a", lw=1.5)
    ax.plot([vin_x, vin_x], [-0.10, y_bot], color="#0f172a", lw=1.5)
    ax.plot([vin_x - 0.22, vin_x + 0.22], [0.10, 0.10], color="#0f172a", lw=1.6)
    ax.plot([vin_x - 0.12, vin_x + 0.12], [-0.10, -0.10], color="#0f172a", lw=2.2)
    ax.text(vin_x - 0.28, 0.0, f"$V_{{IN}}$\n{vin_v:.1f}V", ha="right", va="center", fontsize=8.0, color="#0f172a")
    ax.plot([vin_x, p1_x], [y_bot, y_bot], color="#0f172a", lw=1.5)

    # GND1 symbol on input return
    gnd1_x = -3.65
    ax.scatter([gnd1_x], [y_bot], s=16, color="#0f172a", zorder=5)
    ax.plot([gnd1_x, gnd1_x], [y_bot, y_bot - 0.32], color="#0f172a", lw=1.4)
    ax.plot([gnd1_x - 0.22, gnd1_x + 0.22], [y_bot - 0.32, y_bot - 0.32], color="#0f172a", lw=1.5)
    ax.plot([gnd1_x - 0.14, gnd1_x + 0.14], [y_bot - 0.41, y_bot - 0.41], color="#0f172a", lw=1.4)
    ax.plot([gnd1_x - 0.06, gnd1_x + 0.06], [y_bot - 0.50, y_bot - 0.50], color="#0f172a", lw=1.3)

    # 3. Output Side Circuit (Right: Pin 4 -> Milliammeter mA -> RL -> VCC; Voltmeter VCE across Pins 4-3; Pin 3 -> GND2)
    p4_x = ic_r + 0.28
    vce_x = 2.15
    ma_out_x = 3.05
    rl_r = 4.35
    vcc_x = 5.05

    ax.plot([p4_x, rl_r], [y_top, y_top], color="#0f172a", lw=1.5)
    draw_meter(ax, ma_out_x, y_top, "mA")
    ax.text(ma_out_x, y_top + 0.32, f"$I_C$={ic_ma:.1f}mA", ha="center", va="bottom", fontsize=7.4, color="#b91c1c")
    draw_resistor_h(ax, ma_out_x + 0.24, rl_r, y_top, label=f"$R_L$ ({rl_ohm:.0f}$\\Omega$)", fontsize=7.8)
    ax.plot([rl_r, vcc_x], [y_top, y_top], color="#0f172a", lw=1.5)

    # Voltmeter VCE across Pins 4 and 3
    ax.plot([vce_x, vce_x], [y_top, y_bot], color="#0f172a", lw=1.3)
    ax.scatter([vce_x, vce_x], [y_top, y_bot], s=16, color="#0f172a", zorder=5)
    draw_meter(ax, vce_x, 0.0, "V")
    ax.text(vce_x + 0.30, 0.0, f"$V_{{CE}}$\n{vce_v:.2f}V", ha="left", va="center", fontsize=7.4, color="#b91c1c")

    # Output Variable DC Supply VCC
    ax.plot([vcc_x, vcc_x], [y_top, 0.10], color="#0f172a", lw=1.5)
    ax.plot([vcc_x, vcc_x], [-0.10, y_bot], color="#0f172a", lw=1.5)
    ax.plot([vcc_x - 0.22, vcc_x + 0.22], [0.10, 0.10], color="#0f172a", lw=1.6)
    ax.plot([vcc_x - 0.12, vcc_x + 0.12], [-0.10, -0.10], color="#0f172a", lw=2.2)
    ax.text(vcc_x + 0.28, 0.0, f"$V_{{CC}}$\n{vcc_v:.1f}V", ha="left", va="center", fontsize=8.0, color="#0f172a")
    ax.plot([p4_x, vcc_x], [y_bot, y_bot], color="#0f172a", lw=1.5)

    # GND2 symbol on output return
    gnd2_x = 3.65
    ax.scatter([gnd2_x], [y_bot], s=16, color="#0f172a", zorder=5)
    ax.plot([gnd2_x, gnd2_x], [y_bot, y_bot - 0.32], color="#0f172a", lw=1.4)
    ax.plot([gnd2_x - 0.22, gnd2_x + 0.22], [y_bot - 0.32, y_bot - 0.32], color="#0f172a", lw=1.5)
    ax.plot([gnd2_x - 0.14, gnd2_x + 0.14], [y_bot - 0.41, y_bot - 0.41], color="#0f172a", lw=1.4)
    ax.plot([gnd2_x - 0.06, gnd2_x + 0.06], [y_bot - 0.50, y_bot - 0.50], color="#0f172a", lw=1.3)

    ax.set_xlim(-5.75, 5.75)
    ax.set_ylim(-1.75, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


# -------------------------------------------------------------------------
# Physics Model: PC817 GaAs IR LED + Silicon NPN Phototransistor
# -------------------------------------------------------------------------
def solve_input_led(vin_v, rin_ohm):
    # GaAs IR LED Shockley equation: IF = IS * (exp(VF / (n*VT)) - 1), VF ~ 1.15 - 1.25 V at 5-20 mA
    is_led = 2.5e-14  # A
    n_vt = 0.044      # V (n ~ 1.7 at 300 K)
    if vin_v <= 0.0:
        return 0.0, 0.0
    # Solve Vin = VF + IF * RIN via Newton-Raphson
    vf = min(1.2, vin_v * 0.8)
    for _ in range(35):
        if_a = is_led * (np.exp(np.clip(vf / n_vt, -30, 40)) - 1.0)
        f = vf + if_a * rin_ohm - vin_v
        df = 1.0 + (if_a / n_vt) * rin_ohm
        vf = max(0.0, vf - f / df)
    if_ma = is_led * (np.exp(np.clip(vf / n_vt, -30, 40)) - 1.0) * 1e3
    return float(vf), float(if_ma)


def compute_ic_from_if_and_vce(if_ma, vce_v, ctr_nominal_pct=130.0):
    # Realistic PC817 CTR(IF) bell-shaped efficiency factor (peaks around IF = 5-10 mA)
    if if_ma <= 0.0:
        return 0.0
    eff_shape = 0.78 + 0.24 * np.exp(-((np.log(max(if_ma, 0.2) / 6.5)) ** 2) / 1.8)
    ctr_eff = (ctr_nominal_pct / 100.0) * eff_shape
    ic_sat_target = ctr_eff * if_ma  # Active region collector current at VCE = 5 V

    # Output characteristic knee (VCE_sat ~ 0.2 V) + slight Early slope (VA ~ 65 V)
    vce_pos = max(vce_v, 0.0)
    active_factor = np.tanh(vce_pos / 0.22) * (1.0 + (vce_pos - 5.0) / 70.0)
    return float(max(ic_sat_target * active_factor, 0.0))


def solve_output_operating_point(vcc_v, rl_ohm, if_ma, ctr_nominal_pct=130.0):
    if vcc_v <= 0.0 or if_ma <= 0.0:
        return float(max(vcc_v, 0.0)), 0.0
    # Load line: VCE = VCC - IC * RL -> solve via bisection on VCE in [0, VCC]
    lo, hi = 0.0, vcc_v
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ic_ma = compute_ic_from_if_and_vce(if_ma, mid, ctr_nominal_pct)
        v_drop = (ic_ma * 1e-3) * rl_ohm
        if mid + v_drop < vcc_v:
            lo = mid
        else:
            hi = mid
    vce_op = 0.5 * (lo + hi)
    ic_op = compute_ic_from_if_and_vce(if_ma, vce_op, ctr_nominal_pct)
    return float(vce_op), float(ic_op)


# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + 3 Curves)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Input Circuit Controls (GaAs LED)")
    vin_v = st.slider("Input DC Supply $V_{\\text{IN}}$ (V)", 0.0, 10.0, 3.5, 0.1)
    rin_ohm = st.slider("Input Series Resistor $R_{\\text{IN}}$ (Ohm)", 100, 1000, 470, 10)

    st.subheader("2. Output Circuit Controls (Phototransistor)")
    vcc_v = st.slider("Output Collector Supply $V_{\\text{CC}}$ (V)", 0.0, 15.0, 6.5, 0.25)
    rl_ohm = st.slider("Collector Load Resistor $R_L$ (Ohm)", 100, 2200, 220, 20)

    st.subheader("3. Device Parameter")
    ctr_nominal_pct = st.slider(
        "Nominal PC817 CTR at $V_{CE}=5\\text{ V}, I_F=5\\text{ mA}$ (%)",
        min_value=60.0,
        max_value=250.0,
        value=130.0,
        step=5.0,
        help="Standard PC817 datasheet CTR grade ranges from 50% to 600% (Rank A/B/C)."
    )

# -------------------------------------------------------------------------
# Compute Live Operating Point & CTR
# -------------------------------------------------------------------------
vf_v, if_ma = solve_input_led(vin_v, rin_ohm)
vce_v, ic_ma = solve_output_operating_point(vcc_v, rl_ohm, if_ma, ctr_nominal_pct)
ctr_live_pct = (ic_ma / if_ma) * 100.0 if if_ma > 1e-4 else 0.0
in_active_region = vce_v >= 1.0

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input LED ($V_F$, $I_F$)", f"{vf_v:.2f} V | {if_ma:.2f} mA")
    m2.metric("Output ($V_{CE}$, $I_C$)", f"{vce_v:.2f} V | {ic_ma:.2f} mA")
    m3.metric("Measured CTR ($I_C / I_F$)", f"{ctr_live_pct:.1f} %")
    m4.metric("Operating Region", "Active (Flat)" if in_active_region else "Saturation ($V_{CE} < 1$ V)")

    if in_active_region:
        st.success(
            f"**Phototransistor in Active Region ($V_{{CE}} = {vce_v:.2f}$ V):** "
            f"$\\text{{CTR}} = \\frac{{I_C}}{{I_F}} \\times 100\\% = \\frac{{{ic_ma:.2f}\\text{{ mA}}}}{{{if_ma:.2f}\\text{{ mA}}}} \\times 100\\% = \\mathbf{{{ctr_live_pct:.1f}\\%}}$."
        )
    else:
        st.warning(
            f"**Phototransistor Near Saturation ($V_{{CE}} = {vce_v:.2f}$ V < 1.0 V):** "
            f"Increase $V_{{CC}}$ or reduce $R_L$ so $V_{{CE}} \\approx 5\\text{{ V}}$ (flat active region) for accurate CTR measurement."
        )

    schem_col, curve_col = st.columns([1.15, 1.35], gap="medium")

    with schem_col:
        st.markdown("**PC817 Experimental Circuit Diagram**")
        sch_buf = draw_pc817_schematic(vin_v, vcc_v, rin_ohm, rl_ohm, vf_v, if_ma, vce_v, ic_ma)
        st.image(sch_buf, use_container_width=True)
        st.caption(
            "Complete galvanic isolation between Input LED (Pins 1–2) and Output Phototransistor (Pins 3–4)."
        )

    with curve_col:
        st.markdown("**Characteristic Curves (Input, Output & Transfer)**")
        c_tab1, c_tab2, c_tab3 = st.tabs([
            "1. Input (VF vs. IF)",
            "2. Output (VCE vs. IC)",
            "3. Transfer (IF vs. IC & CTR)"
        ])

        with c_tab1:
            vf_arr = np.linspace(0.0, 1.45, 250)
            if_arr = 2.5e-14 * (np.exp(np.clip(vf_arr / 0.044, -30, 40)) - 1.0) * 1e3
            valid_in = if_arr <= 25.0
            fig_in, ax_in = plt.subplots(figsize=(6.4, 3.8))
            ax_in.plot(vf_arr[valid_in], if_arr[valid_in], color="#1f77b4", lw=2.2, label="GaAs IR LED ($I_F$ vs. $V_F$)")
            ax_in.scatter([vf_v], [if_ma], color="#d62728", zorder=6, s=55, label=f"Operating Point ({vf_v:.2f} V, {if_ma:.2f} mA)")
            style_cro_axes(ax_in, title="Input Characteristics: Forward Current IF vs. Voltage VF", xlabel="LED Forward Voltage VF (V)", ylabel="LED Forward Current IF (mA)")
            ax_in.legend(loc="upper left", fontsize=7.6)
            plt.tight_layout()
            st.pyplot(fig_in, clear_figure=True)
            plt.close(fig_in)

        with c_tab2:
            vce_arr = np.linspace(0.0, 12.0, 250)
            fig_out, ax_out = plt.subplots(figsize=(6.4, 3.8))
            for if_step, col_s in [(2.0, "#9467bd"), (5.0, "#2ca02c"), (10.0, "#1f77b4"), (15.0, "#ff7f0e")]:
                ic_curve = [compute_ic_from_if_and_vce(if_step, vc, ctr_nominal_pct) for vc in vce_arr]
                ax_out.plot(vce_arr, ic_curve, color=col_s, lw=1.9, label=f"$I_F = {if_step:.0f}$ mA")
            ax_out.axvline(5.0, color="#64748b", ls="--", lw=1.0, label="$V_{CE} = 5$ V (CTR Test Bias)")
            ax_out.scatter([vce_v], [ic_ma], color="#d62728", zorder=6, s=55, label=f"Live ($V_{{CE}}={vce_v:.1f}$V, $I_C={ic_ma:.1f}$mA)")
            style_cro_axes(ax_out, title="Output Characteristics: Collector Current IC vs. VCE", xlabel="Collector-Emitter Voltage VCE (V)", ylabel="Collector Current IC (mA)")
            ax_out.legend(loc="upper left", fontsize=7.2, ncol=2)
            plt.tight_layout()
            st.pyplot(fig_out, clear_figure=True)
            plt.close(fig_out)

        with c_tab3:
            if_sweep = np.linspace(0.0, 20.0, 200)
            ic_sweep = [compute_ic_from_if_and_vce(i_val, 5.0, ctr_nominal_pct) for i_val in if_sweep]
            fig_tr, ax_tr = plt.subplots(figsize=(6.4, 3.8))
            ax_tr.plot(if_sweep, ic_sweep, color="#2ca02c", lw=2.2, label="Transfer Curve ($I_C$ vs. $I_F$ at $V_{CE} = 5$ V)")
            ic_at_5v = compute_ic_from_if_and_vce(if_ma, 5.0, ctr_nominal_pct)
            ax_tr.scatter([if_ma], [ic_at_5v], color="#d62728", zorder=6, s=55, label=f"Point ($I_F={if_ma:.1f}$mA, $I_C={ic_at_5v:.1f}$mA)")
            style_cro_axes(ax_tr, title="Transfer Characteristics: IC vs. IF at VCE = 5 V", xlabel="LED Forward Current IF (mA)", ylabel="Collector Current IC (mA)")
            ax_tr.legend(loc="upper left", fontsize=7.6)
            plt.tight_layout()
            st.pyplot(fig_tr, clear_figure=True)
            plt.close(fig_tr)

    # ---------------------------------------------------------------------
    # Secondary Tabs: All 3 Observation Tables + Lab Manual Theory
    # ---------------------------------------------------------------------
    tab_obs1, tab_obs2, tab_obs3, tab_manual = st.tabs([
        "Table 1: Input Characteristics",
        "Table 2: Output Characteristics",
        "Table 3: Transfer Characteristics & CTR",
        "Aim, Theory & Procedure"
    ])

    with tab_obs1:
        st.markdown("**Input Characteristics Observation Table ($V_F$ vs. $I_F$ up to 20 mA):**")
        if_samples = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]
        rows_in = []
        for i_val in if_samples:
            vf_val = 0.0 if i_val == 0 else float(0.044 * np.log((i_val * 1e-3) / 2.5e-14 + 1.0))
            rows_in.append({
                "LED Forward Voltage VF (V)": f"{vf_val:.3f}",
                "LED Forward Current IF (mA)": f"{i_val:.1f}",
            })
        st.table(rows_in)

    with tab_obs2:
        st.markdown("**Output Characteristics Observation Table ($V_{CE}$ vs. $I_C$ at constant $I_F = 5\\text{ mA}$):**")
        vce_samples = [0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0]
        rows_out = []
        for vc_val in vce_samples:
            ic_val = compute_ic_from_if_and_vce(5.0, vc_val, ctr_nominal_pct)
            rows_out.append({
                "Phototransistor Collector-Emitter Voltage VCE (V)": f"{vc_val:.1f}",
                "Phototransistor Collector Current IC (mA)": f"{ic_val:.2f}",
            })
        st.table(rows_out)

    with tab_obs3:
        st.markdown("**Transfer Characteristics Observation Table ($I_F$ from 1 mA to 20 mA at $V_{CE} = 5\\text{ V}$):**")
        rows_tr = []
        ctr_values = []
        for if_int in range(1, 21):
            ic_val = compute_ic_from_if_and_vce(float(if_int), 5.0, ctr_nominal_pct)
            ctr_val = (ic_val / float(if_int)) * 100.0
            ctr_values.append(ctr_val)
            rows_tr.append({
                "LED Forward Current IF (mA)": f"{if_int}",
                "Phototransistor Collector Current IC (mA)": f"{ic_val:.2f}",
                "CTR (%)": f"{ctr_val:.1f} %",
            })
        st.info(f"**Mean Current Transfer Ratio (CTR) across $1\\text{{--}}20\\text{{ mA}}$ at $V_{{CE}} = 5\\text{{ V}}$:** **{np.mean(ctr_values):.1f} %**")
        st.table(rows_tr)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Working Formula

            * **Components Needed:** Optocoupler (**PC817**), DC power supplies (2 nos: $V_{{IN}}$ and $V_{{CC}}$), current-limiting and load resistors, digital multimeters (voltmeters for $V_F$ and $V_{{CE}}$), milliammeters (2 nos for $I_F$ and $I_C$).

            * **Construction & Principle:**
              * **Input Side (Pins 1 & 2):** Gallium Arsenide (GaAs) Infrared LED emitting infrared radiation proportional to forward current $I_F$.
              * **Output Side (Pins 3 & 4):** Silicon NPN Phototransistor whose collector current $I_C$ is controlled optically by the incident infrared flux without any electrical connection between input and output circuits.

            * **Current Transfer Ratio (CTR):**
              $$\\text{{CTR}} = \\frac{{I_C}}{{I_F}} \\times 100\\%$$
              Measured in the **active region** ($V_{{CE}} = 5\\text{{ V}}$), where the output curves are flat and $I_C$ depends primarily on $I_F$.
            """
        )