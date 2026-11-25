import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng, clip_opamp

st.set_page_config(page_title="Exp 2: Op-Amp Differentiator", layout="wide")
render_sidebar_header()

st.title("Experiment 2: Practical Op-Amp Differentiator ($\mu$A741)")
st.markdown(
    "Design and construct a practical differentiator using **$\mu$A741** for **sine wave** and **square wave** inputs, "
    "and study the output waveform and frequency response across different frequencies (*Gayakwad, Section 6.14*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Practical Differentiator Schematic
# -------------------------------------------------------------------------
def draw_differentiator_schematic(r1_ohm, c1_uf, rf_kohm, cf_nf, rom_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.4, fontsize=9.2)
        
        # Input source -> R1 -> C1 -> Summing Node
        d.add(elm.SourceSin().label('$v_{in}$', loc='left'))
        d.add(elm.Line().right().length(0.5))
        d.add(elm.Resistor().right().label(f'$R_1$\n{r1_ohm:.0f} Ω', loc='top'))
        d.add(elm.Capacitor().right().label(f'$C_1$\n{c1_uf:.3f} µF', loc='top'))
        sum_node = d.add(elm.Dot())
        
        # Op-Amp (Inverting terminal at top, Non-inverting at bottom)
        op = d.add(elm.Opamp(leads=True).anchor('in1').label('$\mu$A741', loc='center', fontsize=8.5))
        
        # Power supply labels
        d.add(elm.Line().up().at(op.vd).length(0.45).label(f'+{vcc:.0f}V (7)', loc='top', fontsize=8))
        d.add(elm.Line().down().at(op.vs).length(0.45).label(f'-{vcc:.0f}V (4)', loc='bottom', fontsize=8))
        
        # Non-inverting terminal with Offset Minimizing Resistor R_OM = R_F
        d.add(elm.Line().left().at(op.in2).length(0.5))
        d.add(elm.Resistor().down().label(f'$R_{{OM}}$\n{rom_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Ground())
        
        # Feedback branch: RF in parallel with CF
        d.add(elm.Line().up().at(sum_node.center).length(1.2))
        rf_start = d.here
        d.add(elm.Resistor().right().tox(op.out).label(f'$R_F$\n{rf_kohm:.1f} kΩ', loc='top'))
        d.add(elm.Line().down().toy(op.out))
        out_node = d.add(elm.Dot())
        
        # CF parallel branch above RF
        d.add(elm.Line().up().at(rf_start).length(1.05))
        d.add(elm.Capacitor().right().tox(op.out).label(f'$C_F$\n{cf_nf:.2f} nF', loc='top'))
        d.add(elm.Line().down().length(1.05))
        
        # Output terminal
        d.add(elm.Line().right().at(out_node.center).length(0.9).label('$v_o$ (6)', loc='right'))
        
        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO + Bode)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Input Signal Controls")
    wave_type = st.radio(
        "Select Input Waveform",
        ["Sine Wave", "Square Wave"],
        horizontal=True
    )
    freq = st.slider(
        "Input Frequency $f$ (Hz)",
        min_value=50,
        max_value=20000,
        value=500,
        step=50,
        help="True differentiation occurs for f ≤ fa (where fa = 1 / (2π RF C1))."
    )
    vin_pp = st.slider(
        "Input Peak-to-Peak Voltage $V_{in(pp)}$ (V)",
        min_value=0.2,
        max_value=10.0,
        value=2.0,
        step=0.2
    )

    st.subheader("2. Circuit Components (Gayakwad)")
    c1_uf = st.slider(
        "Input Capacitor $C_1$ (µF)",
        min_value=0.005,
        max_value=0.500,
        value=0.100,
        step=0.005,
        format="%.3f"
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kΩ)",
        min_value=0.5,
        max_value=20.0,
        value=1.5,
        step=0.1
    )
    stability_ratio = st.selectbox(
        "Stability Break Ratio ($f_b / f_a = R_F / R_1$)",
        options=[5, 10, 20],
        index=1,
        help="Gayakwad sets R1·C1 = RF·CF with fb ≥ 10·fa (i.e., R1 = RF / 10) to prevent high-frequency noise and instability."
    )
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 6.14 Equations)
# -------------------------------------------------------------------------
C1 = c1_uf * 1e-6
RF = rf_kohm * 1e3
R1 = RF / stability_ratio
r1_ohm = R1
CF = (R1 * C1) / RF  # Gayakwad design condition: R1*C1 = RF*CF
cf_nf = CF * 1e9
ROM = RF             # DC bias path sees only RF since C1 blocks DC
rom_kohm = ROM / 1e3

# Critical break frequencies (Gayakwad Eqs. 6.45 & 6.46)
fa = 1.0 / (2.0 * np.pi * RF * C1)   # Upper limit of differentiation: fa = 1 / (2π RF C1)
fb = 1.0 / (2.0 * np.pi * R1 * C1)   # Gain-limiting break frequency:  fb = 1 / (2π R1 C1) = 1 / (2π RF CF)
max_hf_gain = -RF / R1

# -------------------------------------------------------------------------
# Exact Time-Domain Simulation (2nd-Order Practical Differentiator)
# -------------------------------------------------------------------------
# Transfer function: H(s) = - (RF * C1 * s) / ((1 + R1*C1*s) * (1 + RF*CF*s))
num_cycles = 4
T = 1.0 / freq
t_total = num_cycles * T
tau_max = max(R1 * C1, RF * CF)
t_settle = max(5.0 * tau_max, 2.0 * T)

t_sim = np.linspace(0.0, t_settle + t_total, 5000)
Vp = vin_pp / 2.0

if wave_type == "Sine Wave":
    v_in_full = Vp * np.sin(2 * np.pi * freq * t_sim)
else:
    # Smooth finite-rise-time square wave (~0.5% of period) so RC spike width is resolved cleanly
    v_in_full = Vp * np.tanh(80.0 * np.sin(2 * np.pi * freq * t_sim))

# Denominator: (R1*C1)*(RF*CF)*s^2 + (R1*C1 + RF*CF)*s + 1
den = [R1 * C1 * RF * CF, R1 * C1 + RF * CF, 1.0]
num = [-RF * C1, 0.0]
sys_diff = signal.TransferFunction(num, den)
_, v_out_full, _ = signal.lsim(sys_diff, U=v_in_full, T=t_sim)

# Slice steady-state display window [t_settle, t_settle + t_total] shifted to 0
mask = t_sim >= t_settle
t_disp = t_sim[mask] - t_settle
vin_disp = v_in_full[mask]
vout_ideal = v_out_full[mask]
vout_disp, vsat = clip_opamp(vout_ideal, vcc)

# -------------------------------------------------------------------------
# Right Display Panel: Always-Visible Circuit Diagram + Live Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Differentiation Limit ($f_a$)", format_eng(fa, "Hz"))
    m2.metric("Break Frequency ($f_b$)", format_eng(fb, "Hz"))
    m3.metric("Stabilizing $R_1$ / $C_F$", f"{r1_ohm:.0f} Ω / {cf_nf:.2f} nF")
    m4.metric("Output $V_{o(pp)}$", f"{np.ptp(vout_disp):.2f} V")

    # Operating regime status banner
    if freq <= fa:
        st.success(
            f"**True Differentiation Region ($f = {freq}\\text{{ Hz}} \\le f_a = {fa:.1f}\\text{{ Hz}}$):** "
            f"Gain rises at **$+20\\text{{ dB/decade}}$** ($v_o \\approx -R_F C_1 \\frac{{dv_{{in}}}}{{dt}}$). "
            f"{'Sine wave differentiates into an inverted cosine wave (-90° shift).' if wave_type == 'Sine Wave' else 'Square wave edges differentiate into sharp positive and negative spikes.'}"
        )
    elif freq <= fb:
        st.warning(
            f"**Transition / Gain-Limited Region ($f_a < f \\le f_b$):** "
            f"Input frequency ($f = {freq}\\text{{ Hz}}$) exceeds the upper differentiation limit $f_a = {fa:.1f}\\text{{ Hz}}$. "
            f"Series resistor $R_1$ limits gain to $-R_F/R_1 = {max_hf_gain:.1f}$, rounding off sharp derivatives."
        )
    else:
        st.error(
            f"**High-Frequency Roll-Off Region ($f = {freq}\\text{{ Hz}} > f_b = {fb:.1f}\\text{{ Hz}}$):** "
            f"Feedback capacitor $C_F$ bypasses $R_F$, causing gain to roll off at $-20\\text{{ dB/decade}}$ (acting as an integrator above $f_b$)."
        )

    if np.max(np.abs(vout_ideal)) > vsat:
        st.info(
            f"**Saturation Clipping Active:** Peak derivative ({np.max(np.abs(vout_ideal)):.2f} V) reaches "
            f"$\\pm V_{{sat}} = \\pm {vsat:.1f}\\text{{ V}}$ (expected for sharp square-wave transitions)."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad*)**")
        fig_sch = draw_differentiator_schematic(r1_ohm, c1_uf, rf_kohm, cf_nf, rom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Design Constraint:** $R_1 C_1 = R_F C_F = {R1*C1*1e6:.2f}\\text{{ µs}}$ and "
            f"$R_{{OM}} = R_F = {rom_kohm:.1f}\\text{{ k}}\\Omega$."
        )

    with cro_col:
        st.markdown("**Input / Output Waveforms (CRO)**")
        fig_cro, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Input
        ax_in.plot(t_ms, vin_disp, color="#1f77b4", linewidth=2, label=f"Input $v_{{in}}(t)$ ({wave_type})")
        style_cro_axes(
            ax_in,
            title=f"CH1: Input ($f = {freq}$ Hz, $V_{{in(pp)}} = {vin_pp:.1f}$ V)",
            xlabel="",
            ylabel="$v_{in}$ (V)"
        )
        ax_in.set_ylim(-Vp * 1.35, Vp * 1.35)
        ax_in.legend(loc="upper right", fontsize=8.5)

        # Channel 2: Output
        ax_out.plot(t_ms, vout_disp, color="#d62728", linewidth=2.2, label="Output $v_o(t)$")
        ax_out.axhline(vsat, color="#888888", linestyle=":", linewidth=1, label=f"$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V")
        ax_out.axhline(-vsat, color="#888888", linestyle=":", linewidth=1)
        y_lim = max(np.max(np.abs(vout_disp)) * 1.35, 0.5)
        ax_out.set_ylim(-y_lim, y_lim)
        style_cro_axes(
            ax_out,
            title=f"CH2: Differentiator Output ($V_{{o(pp)}} = {np.ptp(vout_disp):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # Secondary tabs below for Bode Plot & Theory
    tab_bode, tab_theory = st.tabs([
        "📉 Frequency Response (Bode Plot)",
        "📖 Theory & Design Procedure (Gayakwad)"
    ])

    with tab_bode:
        f_min = max(1.0, fa / 50.0)
        f_max = max(1e5, fb * 50.0)
        f_sweep = np.logspace(np.log10(f_min), np.log10(f_max), 600)
        
        # Gayakwad Eq. (6.44): |A_v| = (f / f_a) / sqrt((1 + (f/f_b)^2) * (1 + (f/f_b)^2)) = (f / f_a) / (1 + (f/f_b)^2)
        gain_mag = (f_sweep / fa) / (1.0 + (f_sweep / fb) ** 2)
        gain_db = 20.0 * np.log10(gain_mag)
        # Phase = 180 deg (inversion) + 90 deg (zero at origin) - 2 * arctan(f / f_b)
        phase_deg = 270.0 - 2.0 * np.degrees(np.arctan(f_sweep / fb))

        fig_bode, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)

        ax_mag.semilogx(f_sweep, gain_db, color="#2ca02c", linewidth=2.2, label="Practical Differentiator Gain $|A_v|$ (dB)")
        ax_mag.axvline(fa, color="#ff7f0e", linestyle="--", linewidth=1.4, label=f"$f_a = {fa:.1f}$ Hz (0 dB / Differentiation limit)")
        ax_mag.axvline(fb, color="#9467bd", linestyle="-.", linewidth=1.4, label=f"$f_b = {fb:.1f}$ Hz (Stabilizing break freq)")
        ax_mag.axvline(freq, color="#d62728", linestyle="-", linewidth=1.6, label=f"Current Input $f = {freq}$ Hz")
        ax_mag.set_ylabel("Gain $|v_o / v_{in}|$ (dB)")
        ax_mag.set_title("Practical Differentiator Frequency Response (Gayakwad Fig. 6-26)", fontweight="bold")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_mag.legend(loc="lower left", fontsize=8.5, ncol=2)

        ax_ph.semilogx(f_sweep, phase_deg, color="#1f77b4", linewidth=2.2, label="Phase Angle $\\phi$ (deg)")
        ax_ph.axvline(freq, color="#d62728", linestyle="-", linewidth=1.6, label=f"Current $f = {freq}$ Hz")
        ax_ph.axhline(270, color="#666666", linestyle=":", linewidth=1, label="270° (-90° Ideal Differentiator)")
        ax_ph.set_ylabel("Phase Angle (degrees)")
        ax_ph.set_xlabel("Frequency $f$ (Hz)")
        ax_ph.set_ylim(80, 285)
        ax_ph.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_ph.legend(loc="lower left", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_bode)
        plt.close(fig_bode)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 6.14*)

            1. **Ideal Differentiator Equation:**
               $$v_o = -R_F C_1 \\frac{{dv_{{in}}}}{{dt}}$$
               For an ideal differentiator, the voltage gain magnitude $|A_v| = 2\\pi f R_F C_1$ increases linearly with frequency at **$+20\\text{{ dB/decade}}$**, which amplifies high-frequency noise and causes high-frequency instability.

            2. **Practical Differentiator Stabilization:**
               To eliminate instability and limit high-frequency noise, Gayakwad adds **$R_1$** in series with $C_1$ and **$C_F$** in parallel with $R_F$, chosen such that $R_1 C_1 = R_F C_F$. The closed-loop gain magnitude becomes:
               $$\\left|\\frac{{v_o}}{{v_{{in}}}}\\right| = \\frac{{f / f_a}}{{1 + (f / f_b)^2}}$$

            3. **Design Frequencies:**
               * **Upper Frequency Limit of Differentiation ($f_a$):**
                 $$f_a = \\frac{{1}}{{2\\pi R_F C_1}} = {fa:.2f}\\text{{ Hz}}$$
               * **Gain-Limiting Break Frequency ($f_b$):**
                 $$f_b = \\frac{{1}}{{2\\pi R_1 C_1}} = \\frac{{1}}{{2\\pi R_F C_F}} = {fb:.2f}\\text{{ Hz}}$$
               * **Condition for True Differentiation:** Good differentiation occurs for all input frequencies:
                 $$f \\le f_a = {fa:.2f}\\text{{ Hz}} \\quad (\\text{{with }} f_b \\ge 10 f_a)$$

            4. **Offset Minimizing Resistor ($R_{{OM}}$):**
               Because input capacitor $C_1$ blocks DC bias current through the input branch, the DC resistance seen at Pin 2 is simply $R_F$. Hence:
               $$R_{{OM}} = R_F = {rom_kohm:.1f}\\text{{ k}}\\Omega$$

            5. **Expected Output Waveforms ($f \\le f_a$):**
               * **Sine Wave Input** $v_{{in}}(t) = V_p \\sin(2\\pi f t)$ $\\longrightarrow$ **Negative Cosine Wave Output:**
                 $$v_o(t) = -2\\pi f R_F C_1 V_p \\cos(2\\pi f t)$$
                 *(Output amplitude increases linearly with frequency $f$ until saturation $\\pm V_{{sat}}$ or break frequency $f_a$ is reached)*
               * **Square Wave Input** $\\longrightarrow$ **Alternating Positive & Negative Spikes** at the falling and rising edges of the square wave.
            """
        )