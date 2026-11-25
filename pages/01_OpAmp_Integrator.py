import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import schemdraw
import schemdraw.elements as elm
from utils import style_cro_axes, format_eng, clip_opamp

st.set_page_config(page_title="Exp 1: Op-Amp Integrator", layout="wide")

st.title("Experiment 1: Practical Op-Amp Integrator ($\mu$A741)")
st.markdown(
    "Design and construct a practical integrator using **$\mu$A741** for **sine wave** and **square wave** inputs, "
    "and study the output waveform and frequency response across different frequencies (*Gayakwad, Section 6.13*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Practical Integrator Schematic
# -------------------------------------------------------------------------
def draw_integrator_schematic(r1_kohm, rf_kohm, cf_uf, rom_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=9.5)
        
        # Input source and R1
        d.add(elm.SourceSin().label('$v_{in}$', loc='left'))
        d.add(elm.Line().right().length(0.7))
        d.add(elm.Resistor().right().label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top'))
        sum_node = d.add(elm.Dot())
        
        # Op-Amp (Inverting terminal at top, Non-inverting at bottom)
        op = d.add(elm.Opamp(leads=True).anchor('in1').label('$\mu$A741', loc='center', fontsize=8.5))
        
        # Power supply labels
        d.add(elm.Line().up().at(op.vd).length(0.45).label(f'+{vcc:.0f}V (7)', loc='top', fontsize=8))
        d.add(elm.Line().down().at(op.vs).length(0.45).label(f'-{vcc:.0f}V (4)', loc='bottom', fontsize=8))
        
        # Non-inverting terminal with Offset Minimizing Resistor R_OM
        d.add(elm.Line().left().at(op.in2).length(0.5))
        d.add(elm.Resistor().down().label(f'$R_{{OM}}$\n{rom_kohm:.2f} kΩ', loc='bottom'))
        d.add(elm.Ground())
        
        # Feedback branch: CF in parallel with RF
        d.add(elm.Line().up().at(sum_node.center).length(1.2))
        cf_start = d.here
        d.add(elm.Capacitor().right().tox(op.out).label(f'$C_F$\n{cf_uf:.3f} µF', loc='top'))
        d.add(elm.Line().down().toy(op.out))
        out_node = d.add(elm.Dot())
        
        # RF parallel branch above CF
        d.add(elm.Line().up().at(cf_start).length(1.05))
        d.add(elm.Resistor().right().tox(op.out).label(f'$R_F$\n{rf_kohm:.1f} kΩ', loc='top'))
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
        ["Square Wave", "Sine Wave"],
        horizontal=True
    )
    freq = st.slider(
        "Input Frequency $f$ (Hz)",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10,
        help="Compare behavior for f < fa (amplifier region) vs. f ≥ 10·fa (true integration region)."
    )
    vin_pp = st.slider(
        "Input Peak-to-Peak Voltage $V_{in(pp)}$ (V)",
        min_value=0.2,
        max_value=10.0,
        value=2.0,
        step=0.2
    )

    st.subheader("2. Circuit Components (Gayakwad)")
    r1_kohm = st.slider("Input Resistor $R_1$ (kΩ)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    rf_ratio = st.selectbox(
        "Feedback Resistor Ratio ($R_F / R_1$)",
        options=[5, 10, 20, 50],
        index=1,
        help="Gayakwad recommends R_F = 10·R_1 to limit DC offset gain while preserving bandwidth."
    )
    cf_uf = st.slider("Feedback Capacitor $C_F$ (µF)", min_value=0.001, max_value=0.500, value=0.010, step=0.001, format="%.3f")
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 6.13 Equations)
# -------------------------------------------------------------------------
R1 = r1_kohm * 1e3
RF = rf_ratio * R1
rf_kohm = RF / 1e3
CF = cf_uf * 1e-6
ROM = (R1 * RF) / (R1 + RF)
rom_kohm = ROM / 1e3

fa = 1.0 / (2.0 * np.pi * RF * CF)   # Lower corner frequency: f_a = 1 / (2π R_F C_F)
fb = 1.0 / (2.0 * np.pi * R1 * CF)   # Unity-gain frequency:   f_b = 1 / (2π R_1 C_F)
dc_gain = -RF / R1

# -------------------------------------------------------------------------
# Exact Time-Domain Simulation (SciPy lsim requires T[0] >= 0)
# -------------------------------------------------------------------------
num_cycles = 4
T = 1.0 / freq
t_total = num_cycles * T
settle_cycles = max(5, int(np.ceil(5.0 / (2.0 * np.pi * fa * T))))
settle_cycles = min(settle_cycles, 80)
t_settle = settle_cycles * T

t_sim = np.linspace(0.0, t_settle + t_total, 4000)
Vp = vin_pp / 2.0

if wave_type == "Sine Wave":
    v_in_full = Vp * np.sin(2 * np.pi * freq * t_sim)
else:
    v_in_full = Vp * signal.square(2 * np.pi * freq * t_sim)

sys_integrator = signal.TransferFunction([-1.0 / (R1 * CF)], [1.0, 1.0 / (RF * CF)])
_, v_out_full, _ = signal.lsim(sys_integrator, U=v_in_full, T=t_sim)

# Slice steady-state display window [t_settle, t_settle + t_total] and shift to 0
mask = t_sim >= t_settle
t_disp = t_sim[mask] - t_settle
vin_disp = v_in_full[mask]
vout_ideal = v_out_full[mask]
vout_disp, vsat = clip_opamp(vout_ideal, vcc)

# -------------------------------------------------------------------------
# Right Display Panel: Always-Visible Circuit Diagram + Live Waveforms
# -------------------------------------------------------------------------
with disp_col:
    # Key Calculated Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lower Corner ($f_a$)", format_eng(fa, "Hz"))
    m2.metric("Unity-Gain ($f_b$)", format_eng(fb, "Hz"))
    m3.metric("Recommended ($10 f_a$)", format_eng(10 * fa, "Hz"))
    m4.metric("Output $V_{o(pp)}$", f"{np.ptp(vout_disp):.2f} V")

    # Operating regime status banner
    if freq < fa:
        st.warning(
            f"**Amplifier Region ($f = {freq}\\text{{ Hz}} < f_a = {fa:.1f}\\text{{ Hz}}$):** "
            f"Capacitor $C_F$ acts nearly as an open circuit. The circuit behaves as a standard inverting amplifier "
            f"with constant gain $-R_F/R_1 = {dc_gain:.1f}$ rather than an integrator."
        )
    elif freq < 10 * fa:
        st.info(
            f"**Transition Region ($f_a \\le f < 10 f_a$):** "
            f"Partial integration occurs with noticeable phase error. Increase $f$ above **{10*fa:.1f} Hz** ($10 f_a$) for accurate integration."
        )
    else:
        st.success(
            f"**True Integration Region ($f = {freq}\\text{{ Hz}} \\ge 10 f_a$):** "
            f"Gain rolls off at **$-20\\text{{ dB/decade}}$** with a **$-90^\\circ$** phase shift (plus $180^\\circ$ inversion = $+90^\\circ$). "
            f"{'Square wave integrates into a triangular wave.' if wave_type == 'Square Wave' else 'Sine wave integrates into a cosine wave.'}"
        )

    if np.max(np.abs(vout_ideal)) > vsat:
        st.error(
            f"**Saturation Clipping:** Ideal output peak ({np.max(np.abs(vout_ideal)):.2f} V) exceeds "
            f"$\\pm V_{{sat}} = \\pm {vsat:.1f}\\text{{ V}}$. Reduce $V_{{in(pp)}}$ or increase $f$."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad*)**")
        fig_sch = draw_integrator_schematic(r1_kohm, rf_kohm, cf_uf, rom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**$\mu$A741 Pinout:** Pin 2 = Inverting (-), Pin 3 = Non-inverting (+), "
            f"Pin 4 = $-V_{{CC}}$, Pin 6 = Output ($v_o$), Pin 7 = $+V_{{CC}}$."
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
            title=f"CH2: Output ($V_{{o(pp)}} = {np.ptp(vout_disp):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # Secondary tabs below the main circuit + waveform view for Bode Plot & Theory
    tab_bode, tab_theory = st.tabs([
        "📉 Frequency Response (Bode Plot)",
        "📖 Theory & Design Procedure (Gayakwad)"
    ])

    with tab_bode:
        f_min = max(1.0, fa / 50.0)
        f_max = max(1e5, fb * 50.0)
        f_sweep = np.logspace(np.log10(f_min), np.log10(f_max), 600)
        
        gain_mag = (RF / R1) / np.sqrt(1.0 + (f_sweep / fa) ** 2)
        gain_db = 20.0 * np.log10(gain_mag)
        phase_deg = 180.0 - np.degrees(np.arctan(f_sweep / fa))

        fig_bode, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)

        ax_mag.semilogx(f_sweep, gain_db, color="#2ca02c", linewidth=2.2, label="Closed-Loop Gain $|A_v|$ (dB)")
        ax_mag.axvline(fa, color="#ff7f0e", linestyle="--", linewidth=1.3, label=f"$f_a = {fa:.1f}$ Hz (-3 dB corner)")
        ax_mag.axvline(10 * fa, color="#9467bd", linestyle="-.", linewidth=1.3, label=f"$10 f_a = {10*fa:.1f}$ Hz (Integration starts)")
        ax_mag.axvline(fb, color="#8c564b", linestyle=":", linewidth=1.3, label=f"$f_b = {fb:.1f}$ Hz (0 dB unity gain)")
        ax_mag.axvline(freq, color="#d62728", linestyle="-", linewidth=1.6, label=f"Current Input $f = {freq}$ Hz")
        ax_mag.set_ylabel("Gain $|v_o / v_{in}|$ (dB)")
        ax_mag.set_title("Practical Integrator Frequency Response (Bode Plot)", fontweight="bold")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_mag.legend(loc="lower left", fontsize=8.5, ncol=2)

        ax_ph.semilogx(f_sweep, phase_deg, color="#1f77b4", linewidth=2.2, label="Phase Shift $\\phi$ (deg)")
        ax_ph.axvline(freq, color="#d62728", linestyle="-", linewidth=1.6, label=f"Current $f = {freq}$ Hz")
        ax_ph.axhline(90, color="#666666", linestyle=":", linewidth=1, label="+90° (Ideal Integrator)")
        ax_ph.set_ylabel("Phase Angle (degrees)")
        ax_ph.set_xlabel("Frequency $f$ (Hz)")
        ax_ph.set_ylim(75, 190)
        ax_ph.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_ph.legend(loc="lower left", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_bode)
        plt.close(fig_bode)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 6.13*)

            1. **Ideal vs. Practical Integrator:**
               An ideal integrator ($v_o = -\\frac{{1}}{{R_1 C_F}} \\int_0^t v_{{in}}\\,dt + v_o(0)$) has infinite DC gain ($f = 0$), which causes any small input offset voltage $V_{{io}}$ to drive the op-amp into saturation. Adding a feedback resistor **$R_F$** across **$C_F$** limits the low-frequency gain to $-R_F / R_1$.

            2. **Closed-Loop Voltage Gain Magnitude:**
               $$\\left|\\frac{{v_o}}{{v_{{in}}}}\\right| = \\frac{{R_F / R_1}}{{\\sqrt{{1 + (f / f_a)^2}}}}$$

            3. **Critical Design Frequencies:**
               * **Lower Frequency Limit ($f_a$):** The $-3\\text{{ dB}}$ corner frequency where gain drops by $3\\text{{ dB}}$ from its DC value:
                 $$f_a = \\frac{{1}}{{2\\pi R_F C_F}} = {fa:.2f}\\text{{ Hz}}$$
               * **Unity-Gain Frequency ($f_b$):** The frequency at which the integrator gain becomes $1$ ($0\\text{{ dB}}$):
                 $$f_b = \\frac{{1}}{{2\\pi R_1 C_F}} = {fb:.2f}\\text{{ Hz}}$$
               * **Condition for True Integration:** Accurate integration with less than $0.5\\%$ amplitude error occurs for all frequencies:
                 $$f \\ge 10 f_a = {10*fa:.2f}\\text{{ Hz}}$$

            4. **Offset Minimizing Resistor ($R_{{OM}}$):**
               To minimize output offset voltage caused by input bias currents ($I_B$), connect $R_{{OM}}$ between Pin 3 and ground:
               $$R_{{OM}} = R_1 \\parallel R_F = \\frac{{R_1 R_F}}{{R_1 + R_F}} = {rom_kohm:.2f}\\text{{ k}}\\Omega$$

            5. **Expected Output Waveforms ($f \\ge 10 f_a$):**
               * **Sine Wave Input** $v_{{in}}(t) = V_p \\sin(2\\pi f t)$ $\\longrightarrow$ **Cosine Wave Output:**
                 $$v_o(t) = \\frac{{V_p}}{{2\\pi f R_1 C_F}} \\cos(2\\pi f t)$$
                 *(Peak-to-peak output inversely proportional to frequency $f$)*
               * **Square Wave Input** ($\\pm V_p$) $\\longrightarrow$ **Triangular Wave Output** with peak-to-peak amplitude:
                 $$V_{{o(pp)}} = \\frac{{V_p}}{{2 f R_1 C_F}}$$
            """
        )
