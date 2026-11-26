import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 17: Three-Input Op-Amp Summing Amplifier", layout="wide")
render_sidebar_header()

st.title("Experiment 17: Three-Input Op-Amp Summing Amplifier ($\\mu$A741)")
st.markdown(
    "Design and construct an Inverting Summing Amplifier** using **$\\mu$A741**, "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 6-1 Three-Input Summing Schematic
# -------------------------------------------------------------------------
def draw_summing_amp_schematic(ra_kohm, rb_kohm, rc_kohm, rf_kohm, rom_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.0)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741', loc='center', fontsize=9))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Output Node Vo and Load Resistor RL ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center
        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$V_o$', loc='right', fontsize=9.5))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} k$\\Omega$', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # --- Summing Junction at Inverting Input (-) ---
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.85))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # Feedback Resistor RF above Op-Amp from summing junction to output
        top_fb_y = sy + 1.35
        d.add(elm.Line().up().at((sx, sy)).toy(top_fb_y))
        d.add(elm.Resistor().right().tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
        d.add(elm.Line().down().toy(fb_y))

        # Three Parallel Input Branches (Va/Ra, Vb/Rb, Vc/Rc) into Summing Junction
        y_a = sy + 0.70
        y_b = sy
        y_c = sy - 0.70

        d.add(elm.Line().up().at((sx, sy)).toy(y_a))
        d.add(elm.Line().down().at((sx, sy)).toy(y_c))

        # Branch A (Top): Va -> Ra
        d.add(elm.Resistor().left().at((sx, y_a)).length(1.95).label(f'$R_a$ ({ra_kohm:.1f}k)', loc='top', fontsize=7.6))
        d.add(elm.Line().left().length(0.35).label('$V_a$', loc='left', fontsize=8.5))

        # Branch B (Middle): Vb -> Rb
        d.add(elm.Resistor().left().at((sx, y_b)).length(1.95).label(f'$R_b$ ({rb_kohm:.1f}k)', loc='top', fontsize=7.6))
        d.add(elm.Line().left().length(0.35).label('$V_b$', loc='left', fontsize=8.5))

        # Branch C (Bottom): Vc -> Rc
        d.add(elm.Resistor().left().at((sx, y_c)).length(1.95).label(f'$R_c$ ({rc_kohm:.1f}k)', loc='top', fontsize=7.6))
        d.add(elm.Line().left().length(0.35).label('$V_c$', loc='left', fontsize=8.5))

        # --- Non-Inverting Bias Compensation Resistor ROM to Ground ---
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.45))
        d.add(elm.Resistor().down().length(1.25).label(f'$R_{{OM}}$\n{rom_kohm:.2f} k$\\Omega$', loc='right', fontsize=7.8))
        d.add(elm.Ground())

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Circuit Mode & Resistors")
    preset_mode = st.selectbox(
        "Quick Design Preset (Gayakwad Sec. 6-2)",
        [
            "Equal-Gain Adder (Ra = Rb = Rc = RF = 10 kOhm)",
            "Averaging Amplifier (Ra = Rb = Rc = 30 kOhm, RF = 10 kOhm)",
            "Custom Weighted Scaling Summer",
        ],
        index=0,
    )

    if preset_mode.startswith("Equal-Gain"):
        ra_def, rb_def, rc_def, rf_def = 10.0, 10.0, 10.0, 10.0
    elif preset_mode.startswith("Averaging"):
        ra_def, rb_def, rc_def, rf_def = 30.0, 30.0, 30.0, 10.0
    else:
        ra_def, rb_def, rc_def, rf_def = 10.0, 20.0, 5.0, 10.0

    rf_kohm = st.slider("Feedback Resistor $R_F$ (kOhm)", 1.0, 100.0, rf_def, 1.0)
    ra_kohm = st.slider("Input Resistor $R_a$ (kOhm)", 1.0, 100.0, ra_def, 1.0)
    rb_kohm = st.slider("Input Resistor $R_b$ (kOhm)", 1.0, 100.0, rb_def, 1.0)
    rc_kohm = st.slider("Input Resistor $R_c$ (kOhm)", 1.0, 100.0, rc_def, 1.0)
    rl_kohm = st.slider("Load Resistor $R_L$ (kOhm)", 1.0, 50.0, 10.0, 1.0)

    st.subheader("2. Three Input Voltages ($V_a, V_b, V_c$)")
    va_in = st.slider("Input Voltage $V_a$ (V)", -5.0, 5.0, 1.0, 0.2)
    vb_in = st.slider("Input Voltage $V_b$ (V)", -5.0, 5.0, 2.0, 0.2)
    vc_in = st.slider("Input Voltage $V_c$ (V)", -5.0, 5.0, -0.5, 0.2)

    st.subheader("3. AC Waveform Mixing & Supply")
    freq_hz = st.slider("AC Signal Frequency $f$ (Hz)", 100, 5000, 1000, 100)
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 6-2, Eqs. 6-1 to 6-4)
# -------------------------------------------------------------------------
vsat = max(vcc - 1.0, 0.5)

gain_a = -(rf_kohm / ra_kohm)
gain_b = -(rf_kohm / rb_kohm)
gain_c = -(rf_kohm / rc_kohm)

vo_ideal = gain_a * va_in + gain_b * vb_in + gain_c * vc_in
vo_actual = float(np.clip(vo_ideal, -vsat, vsat))
saturated = abs(vo_ideal) > vsat

# Bias-compensation resistor ROM = Ra || Rb || Rc || RF (Gayakwad Eq. 6-4)
g_total = (1.0 / ra_kohm) + (1.0 / rb_kohm) + (1.0 / rc_kohm) + (1.0 / rf_kohm)
rom_kohm = 1.0 / g_total

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, CRO Waveforms & Observation Tables
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Output Voltage ($V_o$)", f"{vo_actual:+.2f} V")
    m2.metric("Ideal Sum ($V_{o,\\text{ideal}}$)", f"{vo_ideal:+.2f} V")
    m3.metric("Branch Gains ($-R_F/R_i$)", f"{gain_a:.2f}, {gain_b:.2f}, {gain_c:.2f}")
    m4.metric("Compensation ($R_{OM}$)", f"{rom_kohm:.2f} kOhm")

    if saturated:
        st.warning(
            f"**Op-Amp Output Saturated ($|V_{{o,\\text{{ideal}}}}| = {abs(vo_ideal):.2f}$ V > $V_{{sat}} = {vsat:.1f}$ V):** "
            f"Output clipped at **{vo_actual:+.2f} V**. Reduce input amplitudes or feedback resistor $R_F$ for linear summing."
        )
    else:
        st.success(
            f"**Linear Summing Verified ($V_o = {vo_actual:+.2f}$ V):** "
            f"$V_o = -\\left(\\frac{{{rf_kohm:.0f}}}{{{ra_kohm:.0f}}}({va_in:+.1f}) + "
            f"\\frac{{{rf_kohm:.0f}}}{{{rb_kohm:.0f}}}({vb_in:+.1f}) + "
            f"\\frac{{{rf_kohm:.0f}}}{{{rc_kohm:.0f}}}({vc_in:+.1f})\\right) = {vo_actual:+.2f}\\text{{ V}}$."
        )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_summing_amp_schematic(ra_kohm, rb_kohm, rc_kohm, rf_kohm, rom_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Gayakwad Eq. (6-1):** $V_o = -\\left(\\frac{{R_F}}{{R_a}}V_a + \\frac{{R_F}}{{R_b}}V_b + \\frac{{R_F}}{{R_c}}V_c\\right)$; "
            f"$R_{{OM}} = R_a \\parallel R_b \\parallel R_c \\parallel R_F = {rom_kohm:.2f}\\text{{ k}}\\Omega$."
        )

    with cro_col:
        st.markdown(f"**AC Summing CRO Waveforms ($f = {freq_hz}$ Hz)**")
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        t_ms = t * 1000.0
        w = 2.0 * np.pi * freq_hz

        # Use slider values as signed peak amplitudes of three in-phase sine inputs
        va_t = va_in * np.sin(w * t)
        vb_t = vb_in * np.sin(w * t)
        vc_t = vc_in * np.sin(w * t)
        vo_t = np.clip(gain_a * va_t + gain_b * vb_t + gain_c * vc_t, -vsat, vsat)

        fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        ax_in.plot(t_ms, va_t, color="#1f77b4", linewidth=1.8, label=f"$v_a(t)$ ($V_{{pa}} = {va_in:+.1f}$ V)")
        ax_in.plot(t_ms, vb_t, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"$v_b(t)$ ($V_{{pb}} = {vb_in:+.1f}$ V)")
        ax_in.plot(t_ms, vc_t, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"$v_c(t)$ ($V_{{pc}} = {vc_in:+.1f}$ V)")
        in_lim = max(abs(va_in), abs(vb_in), abs(vc_in), 0.5) * 1.35
        ax_in.set_ylim(-in_lim, in_lim)
        style_cro_axes(
            ax_in,
            title="CH1: Three Input Signals $v_a(t), v_b(t), v_c(t)$",
            xlabel="",
            ylabel="Inputs (V)"
        )
        ax_in.legend(loc="upper right", fontsize=7.5)

        ax_out.plot(t_ms, vo_t, color="#d62728", linewidth=2.2, label=f"Summed Output $v_o(t)$ (Peak = {vo_actual:+.2f} V)")
        out_lim = max(np.max(np.abs(vo_t)) * 1.35, 1.0)
        ax_out.set_ylim(-out_lim, out_lim)
        style_cro_axes(
            ax_out,
            title="CH2: Inverted Weighted Sum Output $v_o(t) = -[(R_F/R_a)v_a + (R_F/R_b)v_b + (R_F/R_c)v_c]$",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Observation Table + Theory & Design Equations
    # ---------------------------------------------------------------------
    tab_obs, tab_theory = st.tabs([
        "Observation Table (DC & AC Summing Verification)",
        "Theory & Design Equations (Gayakwad)"
    ])

    with tab_obs:
        st.markdown(
            f"**Verification Table for $R_a = {ra_kohm:.1f}$ kOhm, $R_b = {rb_kohm:.1f}$ kOhm, "
            f"$R_c = {rc_kohm:.1f}$ kOhm, $R_F = {rf_kohm:.1f}$ kOhm ($\\pm V_{{sat}} = \\pm {vsat:.1f}$ V):**"
        )
        test_cases = [
            (va_in, vb_in, vc_in, "Current Slider Setting"),
            (1.0, 1.0, 1.0, "Equal Positive Inputs"),
            (1.0, 2.0, 3.0, "Ascending Positive Inputs"),
            (2.0, -1.0, 0.5, "Mixed Polarity Inputs"),
            (-1.5, -1.0, -0.5, "All Negative Inputs"),
            (3.0, 3.0, 3.0, "High-Amplitude Test"),
        ]
        rows = []
        for va_t_val, vb_t_val, vc_t_val, note in test_cases:
            v_th = gain_a * va_t_val + gain_b * vb_t_val + gain_c * vc_t_val
            v_pr = float(np.clip(v_th, -vsat, vsat))
            rows.append({
                "Case": note,
                "Va (V)": f"{va_t_val:+.2f}",
                "Vb (V)": f"{vb_t_val:+.2f}",
                "Vc (V)": f"{vc_t_val:+.2f}",
                "Theoretical Vo (V)": f"{v_th:+.2f}",
                "Observed Vo (V)": f"{v_pr:+.2f}",
                "Status": "Saturated" if abs(v_th) > vsat else "Linear"
            })
        st.table(rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (R. A. Gayakwad)

            1. **Virtual Ground & KCL at the Inverting Summing Node:**
               * Because the non-inverting input (`+`) is referenced to ground through bias-compensation resistor $R_{{OM}}$ and open-loop gain $A_{{OL}} \\to \\infty$, the inverting input (`-`) acts as a **virtual ground** ($V_- \\cong 0\\text{{ V}}$) with zero input current ($I_B \\cong 0$).
               * Applying Kirchhoff's Current Law (KCL) at the summing junction:
                 $$I_a + I_b + I_c = I_F \\implies \\frac{{V_a}}{{R_a}} + \\frac{{V_b}}{{R_b}} + \\frac{{V_c}}{{R_c}} = -\\frac{{V_o}}{{R_F}}$$

            2. **Output Voltage Equations (*Gayakwad Eqs. 6-1 to 6-4*):**
               * **Scaling / Weighted Summing Amplifier (*Eq. 6-1*):**
                 $$V_o = -\\left(\\frac{{R_F}}{{R_a}}V_a + \\frac{{R_F}}{{R_b}}V_b + \\frac{{R_F}}{{R_c}}V_c\\right)$$
               * **Equal-Gain Adder ($R_a = R_b = R_c = R_F = R$, *Eq. 6-2*):**
                 $$V_o = -(V_a + V_b + V_c)$$
               * **Averaging Amplifier ($R_a = R_b = R_c = R$ and $R_F = R / 3$, *Eq. 6-3*):**
                 $$V_o = -\\frac{{V_a + V_b + V_c}}{{3}}$$
               * **Input Offset Current Compensation Resistor (*Eq. 6-4*):**
                 $$R_{{OM}} = R_a \\parallel R_b \\parallel R_c \\parallel R_F = {rom_kohm:.2f}\\text{{ k}}\\Omega$$
            """
        )
