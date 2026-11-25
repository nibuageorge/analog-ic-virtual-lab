import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 10: Second-Order Low-Pass Butterworth Filter", layout="wide")
render_sidebar_header()

st.title("Experiment 10: Second-Order Low-Pass Butterworth Filter ($\mu$A741)")
st.markdown(
    "Design and construct a second-order Sallen-Key low-pass Butterworth active filter using **$\mu$A741**, "
    "plot its frequency response curve, and determine the **passband gain ($A_F = 1.586$)**, **high cutoff frequency ($f_H$)**, "
    "and **stopband roll-off rate ($-40\\text{ dB/decade}$)** (*Gayakwad, Section 7.3, Figure 7-3*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 7-3(a) Second-Order Low-Pass Schematic
# -------------------------------------------------------------------------
def draw_second_order_lpf_schematic(r2_kohm, r3_kohm, c2_uf, c3_uf, r1_kohm, rf_kohm, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.2)

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

        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.45).label(f'$R_L$\n{rl_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # --- Upper Negative Feedback Network (R1 & RF) on Inverting (-) Pin ---
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.25
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.45))
        d.add(elm.Line().up().toy(top_y))
        v2_node = d.add(elm.Dot())
        v2_x, v2_y = v2_node.center

        # Feedback Resistor RF to Output fb_tap
        d.add(elm.Resistor().right().at((v2_x, v2_y)).tox(fb_x).label(f'$R_F$\n{rf_kohm:.2f} kΩ', loc='top', fontsize=8))
        d.add(elm.Line().down().at((fb_x, v2_y)).toy(fb_y))

        # Input Ground Resistor R1 to the left
        d.add(elm.Resistor().left().at((v2_x, v2_y)).length(2.15).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top', fontsize=8))
        d.add(elm.Line().down().length(0.35))
        d.add(elm.Ground())

        # --- Lower Sallen-Key 2nd-Order RC Network (R2, R3, C2, C3) on (+) Pin ---
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.65))
        n_plus = d.add(elm.Dot().label('$v_1$', loc='top', fontsize=8.5))
        nx, ny = n_plus.center

        # Shunt Capacitor C3 from (+) node to Ground
        d.add(elm.Capacitor().down().at((nx, ny)).length(1.35).label(f'$C_3$\n{c3_uf:.4f} µF', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Series Resistor R3 going left from (+) node to intermediate node M
        d.add(elm.Resistor().left().at((nx, ny)).length(1.65).label(f'$R_3$\n{r3_kohm:.1f} kΩ', loc='top', fontsize=8))
        n_mid = d.add(elm.Dot())
        mx, my = n_mid.center

        # Series Resistor R2 going left from intermediate node M to AC Source vin
        d.add(elm.Resistor().left().at((mx, my)).length(1.65).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='top', fontsize=8))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, my)).length(0.25))
        d.add(elm.SourceSin().down().length(1.1).label('$v_{in}$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # Positive Feedback Capacitor C2 from intermediate node M under C3 back to Output fb_tap
        fb_bot_y = ny - 1.95
        d.add(elm.Line().down().at((mx, my)).toy(fb_bot_y))
        d.add(elm.Capacitor().right().tox(fb_x).label(f'$C_2$\n{c2_uf:.4f} µF', loc='bottom', fontsize=8))
        d.add(elm.Line().up().toy(fb_y))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Frequency Plot)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Sallen-Key RC Network ($R_2, R_3, C_2, C_3$)")
    r2_kohm = st.slider(
        "Series Resistor $R_2$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=33.0,
        step=0.5,
        help="Gayakwad Example 7-2 default: R2 = R3 = 33 kΩ with C2 = C3 = 0.0047 µF -> fH ≈ 1.03 kHz."
    )
    r3_kohm = st.slider(
        "Series Resistor $R_3$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=33.0,
        step=0.5
    )
    c2_uf = st.slider(
        "Feedback Capacitor $C_2$ (µF)",
        min_value=0.0010,
        max_value=0.1000,
        value=0.0047,
        step=0.0001,
        format="%.4f",
        help="Gayakwad Example 7-2 default: C2 = 0.0047 µF."
    )
    c3_uf = st.slider(
        "Ground Capacitor $C_3$ (µF)",
        min_value=0.0010,
        max_value=0.1000,
        value=0.0047,
        step=0.0001,
        format="%.4f",
        help="Gayakwad Example 7-2 default: C3 = 0.0047 µF."
    )

    st.subheader("2. Butterworth Passband Gain ($R_1, R_F, R_L$)")
    r1_kohm = st.slider(
        "Input Resistor $R_1$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=27.0,
        step=0.5,
        help="Gayakwad Example 7-2 default: R1 = 27 kΩ."
    )
    rf_kohm = st.slider(
        "Feedback Resistor $R_F$ (kΩ)",
        min_value=0.0,
        max_value=100.0,
        value=15.82,
        step=0.1,
        help="For maximally flat Butterworth response (AF = 1.586), set RF = 0.586·R1 (15.82 kΩ for R1 = 27 kΩ)."
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
        step=50
    )
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Exact 2nd-Order Sallen-Key Transfer Function & Butterworth Calculations
# -------------------------------------------------------------------------
R2 = r2_kohm * 1e3
R3 = r3_kohm * 1e3
C2 = c2_uf * 1e-6
C3 = c3_uf * 1e-6
R1 = r1_kohm * 1e3
Rf = rf_kohm * 1e3
vsat = max(vcc - 1.0, 0.5)

af_linear = 1.0 + (Rf / R1)
af_db = 20.0 * np.log10(af_linear)

# Undamped natural frequency wo = 1 / sqrt(R2*R3*C2*C3)
wo = 1.0 / np.sqrt(R2 * R3 * C2 * C3)
fh = wo / (2.0 * np.pi)

# Exact Sallen-Key complex transfer function H(jw) = Vo / Vin:
# H(s) = (AF * wo^2) / (s^2 + s * [1/(R2*C2) + 1/(R3*C2) + (1 - AF)/(R3*C3)] + wo^2)
b_coeff = (1.0 / (R2 * C2)) + (1.0 / (R3 * C2)) + ((1.0 - af_linear) / (R3 * C3))
alpha_damping = b_coeff / wo  # Butterworth condition: alpha = sqrt(2) = 1.414 when R2=R3, C2=C3, AF=1.586

freqs = np.logspace(np.log10(max(fh / 100.0, 1.0)), np.log10(fh * 100.0), 600)
w_arr = 2.0 * np.pi * freqs
s_arr = 1j * w_arr
H_jw = (af_linear * (wo ** 2)) / ((s_arr ** 2) + b_coeff * s_arr + (wo ** 2))

gain_mag = np.abs(H_jw)
gain_db = 20.0 * np.log10(np.maximum(gain_mag, 1e-12))
phase_deg = np.degrees(np.angle(H_jw))

# Gain at f = fH
H_fh = (af_linear * (wo ** 2)) / (((1j * wo) ** 2) + b_coeff * (1j * wo) + (wo ** 2))
gain_at_fh = np.abs(H_fh)
gain_at_fh_db = 20.0 * np.log10(gain_at_fh)

# Measure stopband roll-off rate between 10*fH and 100*fH
w_10, w_100 = 10.0 * wo, 100.0 * wo
g_10_db = 20.0 * np.log10(np.abs((af_linear * (wo ** 2)) / (((1j * w_10) ** 2) + b_coeff * (1j * w_10) + (wo ** 2))))
g_100_db = 20.0 * np.log10(np.abs((af_linear * (wo ** 2)) / (((1j * w_100) ** 2) + b_coeff * (1j * w_100) + (wo ** 2))))
rolloff_db_dec = g_100_db - g_10_db  # ≈ -40.0 dB/decade

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Circuit Diagram, Frequency Response & Tabs
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cutoff Frequency ($f_H$)", format_eng(fh, "Hz"))
    m2.metric("Passband Gain ($A_F$)", f"{af_linear:.3f} ({af_db:+.2f} dB)")
    m3.metric("Gain at $f_H$", f"{gain_at_fh:.3f} ({gain_at_fh_db:+.2f} dB)")
    m4.metric("Stopband Roll-Off Rate", f"{rolloff_db_dec:.2f} dB/dec")

    if alpha_damping <= 0.0:
        st.error(
            f"**Unstable / Oscillating ($A_F = {af_linear:.2f} \\ge 3.00$):** "
            f"Positive feedback through $C_2$ exceeds damping ($\\alpha = {alpha_damping:.2f} \\le 0$). "
            f"Reduce $R_F$ to **$0.586 R_1 = {0.586 * r1_kohm:.2f}\\text{{ k}}\\Omega$** for Butterworth response ($A_F = 1.586$)."
        )
    elif abs(af_linear - 1.586) > 0.08:
        st.warning(
            f"**Non-Butterworth Damping ($A_F = {af_linear:.3f}$, $\\alpha = {alpha_damping:.3f}$):** "
            f"Gayakwad requires **$A_F = 1.586$** ($R_F = 0.586 R_1 = {0.586 * r1_kohm:.2f}\\text{{ k}}\\Omega$) for a maximally flat "
            f"Butterworth response ($-3.01\\text{{ dB}}$ at $f_H$). Current roll-off rate is **{rolloff_db_dec:.2f} dB/decade**."
        )
    else:
        st.success(
            f"**Maximally Flat Butterworth Response ($A_F = {af_linear:.3f} \\approx 1.586$, $\\alpha = {alpha_damping:.3f} \\approx 1.414$):** "
            f"High cutoff **$f_H = {fh:.2f}\\text{{ Hz}}$** ($-3.01\\text{{ dB}}$) with a sharp stopband roll-off of **{rolloff_db_dec:.2f} dB/decade**."
        )

    # Top row: Circuit Diagram (Left) + Frequency Response Curve (Right)
    schem_col, resp_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-3a*)**")
        fig_sch = draw_second_order_lpf_schematic(r2_kohm, r3_kohm, c2_uf, c3_uf, r1_kohm, rf_kohm, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Butterworth Check:** $R_F / R_1 = {rf_kohm/r1_kohm:.3f}$ (Req $0.586$); "
            f"$f_H = \\frac{{1}}{{2\\pi\\sqrt{{R_2 R_3 C_2 C_3}}}} = {fh:.1f}\\text{{ Hz}}$."
        )

    with resp_col:
        st.markdown("**Frequency Response Curve (*Gayakwad Fig. 7-3b*)**")
        fig_freq, (ax_db, ax_lin) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        # Top Plot: Logarithmic Gain Response (dB)
        ax_db.semilogx(freqs, gain_db, color="#1f77b4", linewidth=2.3, label="2nd-Order Gain $20\\log_{10}|v_o/v_{in}|$ (dB)")
        ax_db.axvline(fh, color="#d62728", linestyle="--", linewidth=1.2, label=f"$f_H = {fh:.1f}$ Hz")
        ax_db.axhline(af_db - 3.01, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"$-3$ dB point ({af_db - 3.01:.2f} dB)")
        ax_db.scatter([fh], [gain_at_fh_db], color="#d62728", zorder=5, s=35)
        ax_db.text(fh * 2.1, af_db - 22.0, f"Roll-off = {rolloff_db_dec:.1f} dB/decade", fontsize=8.5, fontweight="bold", color="#d62728")
        ax_db.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_db.set_ylabel("Gain (dB)", fontsize=9, fontweight="bold")
        ax_db.set_title("Logarithmic Gain Response (dB vs. Frequency)", fontsize=9.5, fontweight="bold")
        ax_db.legend(loc="lower left", fontsize=8)

        # Bottom Plot: Linear Voltage Gain Curve
        ax_lin.semilogx(freqs, gain_mag, color="#d62728", linewidth=2.2, label="Voltage Gain $|v_o/v_{in}|$")
        ax_lin.axhline(af_linear, color="#555555", linestyle=":", linewidth=1.1, label=f"$A_F = {af_linear:.3f}$")
        ax_lin.axhline(0.7071 * af_linear, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$0.707 A_F = {0.7071*af_linear:.3f}$")
        ax_lin.axvline(fh, color="#d62728", linestyle="--", linewidth=1.1)
        ax_lin.scatter([fh], [gain_at_fh], color="#d62728", zorder=5, s=35)
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
            s_val = 1j * (2.0 * np.pi * f_val)
            H_val = (af_linear * (wo ** 2)) / ((s_val ** 2) + b_coeff * s_val + (wo ** 2))
            g_val = np.abs(H_val)
            vo_pp_val = min(g_val * vin_pp, 2.0 * vsat)
            g_db_val = 20.0 * np.log10(g_val)
            ph_val = np.degrees(np.angle(H_val))
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
        s_t = 1j * (2.0 * np.pi * f_test)
        H_t = (af_linear * (wo ** 2)) / ((s_t ** 2) + b_coeff * s_t + (wo ** 2))
        g_test = np.abs(H_t)
        phi_test = np.angle(H_t)
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
            ### Theory & Design Procedure (*R. A. Gayakwad, Section 7.3*)

            1. **Butterworth Magnitude Equation (*Gayakwad Eq. 7-4*):**
               A second-order low-pass filter adds an additional $RC$ section ($R_3, C_3$) with positive feedback through $C_2$ to sharpen the stopband roll-off to **$-40\\text{{ dB/decade}}$**:
               $$\\left|\\frac{{v_o}}{{v_{{in}}}}\\right| = \\frac{{A_F}}{{\\sqrt{{1 + (f / f_H)^4}}}}$$

            2. **Key Filter Parameters:**
               * **Passband Gain ($A_F$):** For a maximally flat Butterworth passband response (damping factor $\\alpha = 1.414$), the gain must be fixed at **$1.586$**:
                 $$A_F = 1 + \\frac{{R_F}}{{R_1}} = 1.586 \\implies R_F = 0.586 R_1$$
                 *(Current setting: $A_F = 1 + \\frac{{{rf_kohm:.2f}}}{{{r1_kohm:.1f}}} = {af_linear:.3f}$, or **${af_db:+.2f}\\text{{ dB}}$**.)*
               * **High Cutoff Frequency ($f_H$):**
                 $$f_H = \\frac{{1}}{{2\\pi \\sqrt{{R_2 R_3 C_2 C_3}}}} = \\frac{{1}}{{2\\pi R C}} = {fh:.2f}\\text{{ Hz}}$$
               * **Stopband Roll-Off Rate:**
                 For $f \\gg f_H$, $|v_o / v_{{in}}| \\approx A_F (f_H / f)^2$, yielding a steep attenuation slope of **$-40\\text{{ dB/decade}}$** ($-12\\text{{ dB/octave}}$).

            3. **Design Steps (*Gayakwad Example 7-2*):**
               1. Choose the desired high cutoff frequency $f_H$ (e.g., $1\\text{{ kHz}}$).
               2. Set $R_2 = R_3 = R$ and $C_2 = C_3 = C$. Pick $C \\le 1\\text{{ }}\\mu\\text{{F}}$ (e.g., $0.0047\\text{{ }}\\mu\\text{{F}}$) and calculate $R = \\frac{{1}}{{2\\pi f_H C}} \\approx 33.86\\text{{ k}}\\Omega$ (use $33\\text{{ k}}\\Omega$ standard value).
               3. Choose $R_1$ (e.g., $27\\text{{ k}}\\Omega$) and set $R_F = 0.586 R_1 = 15.82\\text{{ k}}\\Omega$ (using a $20\\text{{ k}}\\Omega$ potentiometer) to guarantee $A_F = 1.586$.
            """
        )