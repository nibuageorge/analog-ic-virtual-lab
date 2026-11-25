import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 9: First-Order Low-Pass Butterworth Filter", layout="wide")
render_sidebar_header()

st.title("Experiment 9: First-Order Low-Pass Butterworth Filter ($\mu$A741)")
st.markdown(
    "Design and construct a first-order low-pass Butterworth active filter using **$\mu$A741**, "
    "plot its frequency response curve, and determine the **passband gain ($A_F$)**, **high cutoff frequency ($f_H$)**, "
    "and **stopband roll-off rate** (*Gayakwad, Section 7.2, Figure 7-2*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 7-2(a) First-Order Low-Pass Schematic
# -------------------------------------------------------------------------
def draw_first_order_lpf_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.4)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Output Node v_o and Load Resistor R_L ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center

        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o$', loc='right', fontsize=9.5))

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} kΩ', loc='bottom', fontsize=8.5))
        d.add(elm.Ground())

        # --- Upper Negative Feedback Network (R1 & RF) on Inverting (-) Pin ---
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.25
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.45))
        d.add(elm.Line().up().toy(top_y))
        v2_node = d.add(elm.Dot().label('$v_2$', loc='top', fontsize=8.5))
        v2_x, v2_y = v2_node.center

        # Feedback Resistor RF to Output fb_tap
        d.add(elm.Resistor().right().at((v2_x, v2_y)).tox(fb_x).label(f'$R_F$\n{rf_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        d.add(elm.Line().down().at((fb_x, v2_y)).toy(fb_y))

        # Input Ground Resistor R1 to the left
        d.add(elm.Resistor().left().at((v2_x, v2_y)).length(1.95).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        d.add(elm.Line().down().length(0.35))
        d.add(elm.Ground())

        # --- Lower RC Low-Pass Network (R & C) on Non-Inverting (+) Pin ---
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.75))
        v1_node = d.add(elm.Dot().label('$v_1$', loc='top', fontsize=8.5))
        v1_x, v1_y = v1_node.center

        # Capacitor C from node v1 to Ground
        d.add(elm.Capacitor().down().at((v1_x, v1_y)).length(1.35).label(f'$C$\n{c_uf:.3f} µF', loc='bottom', fontsize=8.5))
        d.add(elm.Ground())

        # Potentiometer R (20-kΩ pot set at r_kohm) from AC Source v_in to node v1
        d.add(elm.Potentiometer().left().at((v1_x, v1_y)).length(2.15).label(f'$R$ (20-k pot)\nSet at {r_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, v1_y)).length(0.25))
        d.add(elm.SourceSin().down().length(1.1).label('$v_{in}$', loc='left', fontsize=9))
        d.add(elm.Ground())

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Frequency Plot)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Low-Pass RC Network ($R, C$)")
    r_kohm = st.slider(
        "Filter Potentiometer $R$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=15.9,
        step=0.1,
        help="Gayakwad Fig. 7-2(a) default: 20-kΩ pot set at 15.9 kΩ with C = 0.01 µF -> fH ≈ 1.00 kHz."
    )
    c_uf = st.slider(
        "Filter Capacitor $C$ (µF)",
        min_value=0.001,
        max_value=0.200,
        value=0.010,
        step=0.001,
        format="%.3f",
        help="Gayakwad Fig. 7-2(a) default: 0.01 µF."
    )

    st.subheader("2. Passband Gain Network ($R_1, R_F, R_L$)")
    r1_kohm = st.slider(
        "Input Resistor $R_1$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Gayakwad Fig. 7-2(a) default: 10 kΩ."
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kΩ)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Passband gain AF = 1 + RF / R1 (default 10 kΩ -> AF = 2.00 or +6.02 dB)."
    )
    rl_kohm = st.slider(
        "Load Resistor $R_L$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    st.subheader("3. Input Test Signal & Supply")
    vin_pp = st.slider("Input Signal $V_{in(pp)}$ (V)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    f_test = st.slider(
        "CRO Test Frequency $f_{test}$ (Hz)",
        min_value=50,
        max_value=20000,
        value=1000,
        step=50,
        help="Adjust to inspect input/output attenuation and phase lag on the CRO tab."
    )
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Filter Calculations (Gayakwad Eq. 7-1b)
# -------------------------------------------------------------------------
R = r_kohm * 1e3
C = c_uf * 1e-6
R1 = r1_kohm * 1e3
Rf = rf_kohm * 1e3
vsat = max(vcc - 1.0, 0.5)

af_linear = 1.0 + (Rf / R1)
af_db = 20.0 * np.log10(af_linear)
fh = 1.0 / (2.0 * np.pi * R * C)

# Frequency array spanning 2 decades below fH to 2 decades above fH
freqs = np.logspace(np.log10(max(fh / 100.0, 1.0)), np.log10(fh * 100.0), 600)
gain_mag = af_linear / np.sqrt(1.0 + (freqs / fh) ** 2)
gain_db = 20.0 * np.log10(gain_mag)
phase_deg = -np.degrees(np.arctan(freqs / fh))

# Measure stopband roll-off rate directly from frequency response between 10*fH and 100*fH
g_10fh_db = 20.0 * np.log10(af_linear / np.sqrt(1.0 + 10.0 ** 2))
g_100fh_db = 20.0 * np.log10(af_linear / np.sqrt(1.0 + 100.0 ** 2))
rolloff_db_dec = g_100fh_db - g_10fh_db  # ≈ -20.0 dB/decade

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Circuit Diagram, Frequency Response & Tabs
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cutoff Frequency ($f_H$)", format_eng(fh, "Hz"))
    m2.metric("Passband Gain ($A_F$)", f"{af_linear:.2f} ({af_db:+.2f} dB)")
    m3.metric("Gain at $f_H$ ($0.707 A_F$)", f"{0.7071 * af_linear:.2f} ({af_db - 3.01:+.2f} dB)")
    m4.metric("Stopband Roll-Off Rate", f"{rolloff_db_dec:.2f} dB/dec")

    st.success(
        f"**First-Order Low-Pass Butterworth Response:** High cutoff frequency **$f_H = {fh:.2f}\\text{{ Hz}}$**, "
        f"passband voltage gain **$A_F = {af_linear:.2f}$** (**${af_db:+.2f}\\text{{ dB}}$**), and stopband roll-off rate of **{rolloff_db_dec:.2f} dB/decade**."
    )

    # Top row: Circuit Diagram (Left) + Frequency Response Curve (Right)
    schem_col, resp_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-2a*)**")
        fig_sch = draw_first_order_lpf_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Design Formulas:** $f_H = \\frac{{1}}{{2\\pi R C}} = {fh:.1f}\\text{{ Hz}}$; "
            f"$A_F = 1 + \\frac{{R_F}}{{R_1}} = {af_linear:.2f}$."
        )

    with resp_col:
        st.markdown("**Frequency Response Curve (*Gayakwad Fig. 7-2b*)**")
        fig_freq, (ax_db, ax_lin) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        # Top Plot: Magnitude Response in dB (Bode Magnitude Plot)
        ax_db.semilogx(freqs, gain_db, color="#1f77b4", linewidth=2.3, label="Gain $20\\log_{10}|v_o/v_{in}|$ (dB)")
        ax_db.axvline(fh, color="#d62728", linestyle="--", linewidth=1.2, label=f"$f_H = {fh:.1f}$ Hz")
        ax_db.axhline(af_db - 3.01, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"$-3$ dB point ({af_db - 3.01:.1f} dB)")
        ax_db.scatter([fh], [af_db - 3.01], color="#d62728", zorder=5, s=35)
        ax_db.text(fh * 2.2, af_db - 11.5, f"Roll-off = {rolloff_db_dec:.1f} dB/decade", fontsize=8.5, fontweight="bold", color="#d62728")
        ax_db.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_db.set_ylabel("Gain (dB)", fontsize=9, fontweight="bold")
        ax_db.set_title("Logarithmic Gain Response (dB vs. Frequency)", fontsize=9.5, fontweight="bold")
        ax_db.legend(loc="lower left", fontsize=8)

        # Bottom Plot: Normalized Linear Voltage Gain (matching Gayakwad Fig. 7-2b)
        ax_lin.semilogx(freqs, gain_mag, color="#d62728", linewidth=2.2, label="Voltage Gain $|v_o/v_{in}|$")
        ax_lin.axhline(af_linear, color="#555555", linestyle=":", linewidth=1.1, label=f"$A_F = {af_linear:.2f}$")
        ax_lin.axhline(0.7071 * af_linear, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$0.707 A_F = {0.7071*af_linear:.2f}$")
        ax_lin.axvline(fh, color="#d62728", linestyle="--", linewidth=1.1)
        ax_lin.scatter([fh], [0.7071 * af_linear], color="#d62728", zorder=5, s=35)
        ax_lin.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_lin.set_xlabel("Frequency $f$ (Hz) [Log Scale]", fontsize=9, fontweight="bold")
        ax_lin.set_ylabel("Gain $|v_o / v_{in}|$", fontsize=9, fontweight="bold")
        ax_lin.set_title(f"Linear Voltage Gain Curve (Passband < {fh:.0f} Hz | Stopband > {fh:.0f} Hz)", fontsize=9.5, fontweight="bold")
        ax_lin.legend(loc="lower left", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_freq)
        plt.close(fig_freq)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Observation Table + CRO Waveforms + Theory
    # ---------------------------------------------------------------------
    tab_obs, tab_cro, tab_theory = st.tabs([
        "📊 Frequency Response Observation Table",
        "🖥️ Time-Domain CRO Waveforms at Test Frequency",
        "📖 Theory & Design Procedure (Gayakwad)"
    ])

    with tab_obs:
        st.markdown(
            f"Tabulated gain and phase readings across frequencies relative to **$f_H = {fh:.2f}\\text{{ Hz}}$** "
            f"with constant input **$V_{{in(pp)}} = {vin_pp:.2f}\\text{{ V}}$**:"
        )
        multipliers = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]
        obs_rows = []
        for mult in multipliers:
            f_val = mult * fh
            g_val = af_linear / np.sqrt(1.0 + (f_val / fh) ** 2)
            vo_pp_val = min(g_val * vin_pp, 2.0 * vsat)
            g_db_val = 20.0 * np.log10(g_val)
            ph_val = -np.degrees(np.arctan(f_val / fh))
            region = "Cutoff (-3 dB)" if np.isclose(mult, 1.0) else ("Passband" if mult < 1.0 else "Stopband")
            obs_rows.append({
                "Frequency Ratio (f / fH)": f"{mult:.2f} fH",
                "Frequency f (Hz)": f"{f_val:.1f}",
                "Input Vin(pp) (V)": f"{vin_pp:.2f}",
                "Output Vo(pp) (V)": f"{vo_pp_val:.2f}",
                "Voltage Gain |Vo/Vin|": f"{g_val:.3f}",
                "Gain (dB)": f"{g_db_val:+.2f}",
                "Phase Shift (deg)": f"{ph_val:.1f}°",
                "Region": region
            })
        st.table(obs_rows)

    with tab_cro:
        T_test = 1.0 / f_test
        t_cro = np.linspace(0.0, 3.0 * T_test, 2000)
        vin_wave = (vin_pp / 2.0) * np.sin(2.0 * np.pi * f_test * t_cro)
        g_test = af_linear / np.sqrt(1.0 + (f_test / fh) ** 2)
        phi_test = -np.arctan(f_test / fh)
        vo_wave = np.clip(g_test * (vin_pp / 2.0) * np.sin(2.0 * np.pi * f_test * t_cro + phi_test), -vsat, vsat)

        fig_cro, ax_cro = plt.subplots(figsize=(8.8, 3.8))
        ax_cro.plot(t_cro * 1000.0, vin_wave, color="#1f77b4", linewidth=2.0, label=f"Input $v_{{in}}(t)$ ($V_{{pp}} = {vin_pp:.2f}$ V)")
        ax_cro.plot(t_cro * 1000.0, vo_wave, color="#d62728", linewidth=2.2,
                    label=f"Output $v_o(t)$ ($|A_v| = {g_test:.2f}$, $\\phi = {np.degrees(phi_test):.1f}^\\circ$)")
        y_cro_lim = max(np.max(np.abs(vin_wave)), np.max(np.abs(vo_wave))) * 1.25
        ax_cro.set_ylim(-y_cro_lim, y_cro_lim)
        style_cro_axes(
            ax_cro,
            title=f"CRO Display at $f_{{test}} = {f_test}$ Hz (Cutoff $f_H = {fh:.1f}$ Hz)",
            xlabel="Time (ms)",
            ylabel="Voltage (V)"
        )
        ax_cro.legend(loc="upper right", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Procedure (*R. A. Gayakwad, Section 7.2, p. 254*)

            1. **Transfer Function (*Gayakwad Eq. 7-1b*):**
               The voltage $v_1$ across capacitor $C$ is $v_1 = \\frac{{v_{{in}}}}{{1 + j 2\\pi f R C}}$, and the non-inverting op-amp amplifies $v_1$ by $A_F = 1 + \\frac{{R_F}}{{R_1}}$:
               $$\\frac{{v_o}}{{v_{{in}}}} = \\frac{{A_F}}{{1 + j(f / f_H)}}$$
               Taking the magnitude gives the frequency response:
               $$\\left|\\frac{{v_o}}{{v_{{in}}}}\\right| = \\frac{{A_F}}{{\\sqrt{{1 + (f / f_H)^2}}}}$$

            2. **Key Filter Parameters:**
               * **Passband Gain ($A_F$):**
                 $$A_F = 1 + \\frac{{R_F}}{{R_1}} = 1 + \\frac{{{rf_kohm:.1f}}}{{{r1_kohm:.1f}}} = {af_linear:.2f} \\quad ({af_db:+.2f}\\text{{ dB}})$$
               * **High Cutoff Frequency ($f_H$):**
                 $$f_H = \\frac{{1}}{{2\\pi R C}} = \\frac{{1}}{{2\\pi ({r_kohm:.1f}\\text{{ k}}\\Omega)({c_uf:.3f}\\text{{ }}\\mu\\text{{F}})}} = {fh:.2f}\\text{{ Hz}}$$
               * **Stopband Roll-Off Rate:**
                 For $f \\gg f_H$, $|v_o / v_{{in}}| \\approx A_F (f_H / f)$, so the gain decreases at a constant rate of **$-20\\text{{ dB/decade}}$** (or $-6\\text{{ dB/octave}}$)[cite: 10].

            3. **Filter Design Steps:**
               1. Choose a desired high cutoff frequency $f_H$ (e.g., $1\\text{{ kHz}}$).
               2. Select a standard value of capacitor $C \\le 1\\text{{ }}\\mu\\text{{F}}$ (e.g., $0.01\\text{{ }}\\mu\\text{{F}}$).
               3. Calculate $R = \\frac{{1}}{{2\\pi f_H C}}$ (e.g., $15.9\\text{{ k}}\\Omega$ using a $20\\text{{ k}}\\Omega$ potentiometer).
               4. Select $R_1$ and $R_F$ according to the required passband gain $A_F = 1 + \\frac{{R_F}}{{R_1}}$.
            """
        )