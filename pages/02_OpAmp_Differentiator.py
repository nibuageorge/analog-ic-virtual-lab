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
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Practical Differentiator Schematic
# -------------------------------------------------------------------------
def draw_differentiator_schematic(r1_ohm, c1_uf, rf_kohm, cf_nf, rom_kohm, vcc):
    r1_label = f"{r1_ohm/1000:.2f} kΩ" if r1_ohm >= 1000 else f"{r1_ohm:.0f} Ω"
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.4, fontsize=9.2)
        
        # Input source -> R1 -> C1 -> Summing Node
        d.add(elm.SourceSin().label('$v_{in}$', loc='left'))
        d.add(elm.Line().right().length(0.5))
        d.add(elm.Resistor().right().label(f'$R_1$\n{r1_label}', loc='top'))
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
    r1_ohm = st.slider(
        "Series Input Resistor $R_1$ (Ω)",
        min_value=10,
        max_value=10000,
        value=150,
        step=10,
        help="Variable from 10 Ω to 10 kΩ (10,000 Ω). Limits high-frequency gain to -RF/R1 and sets break frequency fb1 = 1 / (2π R1 C1)."
    )
    c1_uf = st.slider(
        "Input Capacitor $C_1$ (µF)",
        min_value=0.005,
        max_value=1.000,
        value=0.100,
        step=0.001,
        format="%.3f"
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kΩ)",
        min_value=0.5,
        max_value=20.0,
        value=1.5,
        step=0.1
    )
    auto_cf = st.checkbox(
        "Auto-match $C_F$ ($R_F C_F = R_1 C_1$)",
        value=True,
        help="Gayakwad recommends choosing CF so that RF·CF = R1·C1 (making fb1 = fb2)."
    )
    if auto_cf:
        cf_nf = (r1_ohm * c1_uf * 1e-6 / (rf_kohm * 1e3)) * 1e9
        st.caption(f"Matched Feedback Capacitor: **$C_F = {cf_nf:.2f}\\text{{ nF}}$**")
    else:
        cf_nf = st.slider(
            "Feedback Capacitor $C_F$ (nF)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1
        )
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 6.14 Equations)
# -------------------------------------------------------------------------
R1 = float(r1_ohm)
C1 = c1_uf * 1e-6
RF = rf_kohm * 1e3
CF = cf_nf * 1e-9
ROM = RF             # DC bias path sees only RF since C1 blocks DC
rom_kohm = ROM / 1e3
r1_disp_str = format_eng(R1, "Ω")

# Critical break frequencies (Gayakwad Eqs. 6.45 & 6.46)
fa = 1.0 / (2.0 * np.pi * RF * C1)    # Upper limit of differentiation: fa = 1 / (2π RF C1)
fb1 = 1.0 / (2.0 * np.pi * R1 * C1)   # Input break frequency:          fb1 = 1 / (2π R1 C1)
fb2 = 1.0 / (2.0 * np.pi * RF * CF)   # Feedback break frequency:       fb2 = 1 / (2π RF CF)
fb = min(fb1, fb2)
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
    v_in_full = Vp * np.tanh(80.0 * np.sin(2 * np.pi * freq * t_sim))

den = [R1 * C1 * RF * CF, R1 * C1 + RF * CF, 1.0]
num = [-RF * C1, 0.0]
sys_diff = signal.TransferFunction(num, den)
_, v_out_full, _ = signal.lsim(sys_diff, U=v_in_full, T=t_sim)

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
    m2.metric("Input Break ($f_{b1}$)", format_eng(fb1, "Hz"))
    m3.metric("Feedback Break ($f_{b2}$)", format_eng(fb2, "Hz"))
    m4.metric("Output $V_{o(pp)}$", f"{np.ptp(vout_disp):.2f} V")

    # Operating regime status banner
    if R1 >= RF:
        st.warning(
            f"**High Series Resistance ($R_1 = {r1_disp_str} \\ge R_F = {rf_kohm:.1f}\\text{{ k}}\\Omega$):** "
            f"Because $R_1 \\ge R_F$, $f_{{b1}} \\le f_a$; the circuit attenuates the signal ($|A_v| \\le {abs(max_hf_gain):.2f}$) "
            f"instead of acting as a true differentiator."
        )
    elif freq <= fa and freq <= fb:
        st.success(
            f"**True Differentiation Region ($f = {freq}\\text{{ Hz}} \\le f_a = {fa:.1f}\\text{{ Hz}}$):** "
            f"Gain rises at **$+20\\text{{ dB/decade}}$** ($v_o \\approx -R_F C_1 \\frac{{dv_{{in}}}}{{dt}}$). "
            f"{'Sine wave differentiates into an inverted cosine wave (-90° shift).' if wave_type == 'Sine Wave' else 'Square wave edges differentiate into sharp positive and negative spikes.'}"
        )
    elif freq <= fb:
        st.warning(
            f"**Transition / Gain-Limited Region ($f_a < f \\le f_b$):** "
            f"Input frequency ($f = {freq}\\text{{ Hz}}$) exceeds the upper differentiation limit $f_a = {fa:.1f}\\text{{ Hz}}$. "
            f"Series resistor $R_1 = {r1_disp_str}$ limits gain to $-R_F/R_1 = {max_hf_gain:.2f}$, broadening the derivative."
        )
    else:
        st.error(
            f"**High-Frequency Roll-Off Region ($f = {freq}\\text{{ Hz}} > f_b = {fb:.1f}\\text{{ Hz}}$):** "
            f"Above $f_b$, the circuit rolls off at $-20\\text{{ dB/decade}}$ (acting as an integrator)."
        )

    if np.max(np.abs(vout_ideal)) > vsat:
        st.info(
            f"**Saturation Clipping Active:** Peak derivative ({np.max(np.abs(vout_ideal)):.2f} V) reaches "
            f"$\\pm V_{{sat}} = \\pm {vsat:.1f}\\text{{ V}}$ (increase $R_1$ or reduce $V_{{in(pp)}}$ to avoid clipping)."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad*)**")
        fig_sch = draw_differentiator_schematic(r1_ohm, c1_uf, rf_kohm, cf_nf, rom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Time Constants:** $R_1 C_1 = {format_eng(R1*C1, 's')}$, "
            f"$R_F C_F = {format_eng(RF*CF, 's')}$, $R_{{OM}} = {rom_kohm:.1f}\\text{{ k}}\\Omega$."
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
        f_min = max(1.0, min(fa, fb1, fb2) / 50.0)
        f_max = max(1e5, max(fa, fb1, fb2) * 50.0)
        f_sweep = np.logspace(np.log10(f_min), np.log10(f_max), 600)
        
        # Exact 2nd-order magnitude & phase for arbitrary R1, C1, RF, CF
        gain_mag = (f_sweep / fa) / np.sqrt((1.0 + (f_sweep / fb1) ** 2) * (1.0 + (f_sweep / fb2) ** 2))
        gain_db = 20.0 * np.log10(gain_mag)
        phase_deg = 270.0 - np.degrees(np.arctan(f_sweep / fb1)) - np.degrees(np.arctan(f_sweep / fb2))

        fig_bode, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)

        ax_mag.semilogx(f_sweep, gain_db, color="#2ca02c", linewidth=2.2, label="Practical Differentiator Gain $|A_v|$ (dB)")
        ax_mag.axvline(fa, color="#ff7f0e", linestyle="--", linewidth=1.4, label=f"$f_a = {fa:.1f}$ Hz (0 dB / Differentiation limit)")
        ax_mag.axvline(fb1, color="#9467bd", linestyle="-.", linewidth=1.4, label=f"$f_{{b1}} = {fb1:.1f}$ Hz ($1 / 2\\pi R_1 C_1$)")
        if not np.isclose(fb1, fb2, rtol=1e-2):
            ax_mag.axvline(fb2, color="#8c564b", linestyle=":", linewidth=1.4, label=f"$f_{{b2}} = {fb2:.1f}$ Hz ($1 / 2\\pi R_F C_F$)")
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

            2. **Role of Series Resistor $R_1$ and Feedback Capacitor $C_F$:**
               * Adding **$R_1$** ({r1_disp_str}) in series with $C_1$ limits the high-frequency voltage gain to $-R_F / R_1 = {max_hf_gain:.2f}$ and sets the break frequency $f_{{b1}} = \\frac{{1}}{{2\\pi R_1 C_1}} = {fb1:.2f}\\text{{ Hz}}$.
               * Adding **$C_F$** ({cf_nf:.2f} nF) across $R_F$ introduces a second break frequency $f_{{b2}} = \\frac{{1}}{{2\\pi R_F C_F}} = {fb2:.2f}\\text{{ Hz}}$ to suppress oscillation above $f_{{b2}}$.
               * **Closed-Loop Gain Magnitude:**
                 $$\\left|\\frac{{v_o}}{{v_{{in}}}}\\right| = \\frac{{f / f_a}}{{\\sqrt{{(1 + (f / f_{{b1}})^2)(1 + (f / f_{{b2}})^2)}}}}$$

            3. **Design Frequencies:**
               * **Upper Frequency Limit of Differentiation ($f_a$):**
                 $$f_a = \\frac{{1}}{{2\\pi R_F C_1}} = {fa:.2f}\\text{{ Hz}}$$
               * **Condition for True Differentiation:** Good differentiation occurs for all input frequencies:
                 $$f \\le f_a = {fa:.2f}\\text{{ Hz}} \\quad (\\text{{with }} f_b \\ge 10 f_a)$$

            4. **Offset Minimizing Resistor ($R_{{OM}}$):**
               Because input capacitor $C_1$ blocks DC bias current through the input branch, the DC resistance seen at Pin 2 is $R_F$:
               $$R_{{OM}} = R_F = {rom_kohm:.1f}\\text{{ k}}\\Omega$$
            """
        )
