import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 13: Narrow Band-Pass Filter", layout="wide")
render_sidebar_header()

st.title("Experiment 13: Multiple-Feedback Narrow Band-Pass Filter ($\mu$A741)")
st.markdown(
    "Design and construct a single op-amp multiple-feedback **Narrow Band-Pass Filter** using **$\mu$A741**, "
    "plot its frequency response curve to determine the **bandwidth ($BW = f_H - f_L$)** and **center-frequency gain ($A_F$)**, "
    "and repeat the experiment for a second center frequency ($f_C'$) at the **same gain ($A_F$)** "
    "(*Gayakwad, Section 7.8, Figure 7-13*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 7-13(a) Narrow Band-Pass Schematic
# -------------------------------------------------------------------------
def draw_narrow_bpf_schematic(r1_kohm, r2_kohm, r3_kohm, c_uf, rl_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.2)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Offset-Minimizing Resistor R3 on Non-Inverting (+) Pin ---
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.55))
        d.add(elm.Resistor().down().length(1.15).label(f'$R_3$\n{r3_kohm:.2f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

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

        # --- Inverting (-) Path: Capacitor C1 & Inner Feedback Resistor R3 ---
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.5))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # Inner Feedback Loop (R3) above Op-Amp from sum_node to Output fb_tap
        r3_y = sy + 1.05
        d.add(elm.Line().up().at((sx, sy)).toy(r3_y))
        d.add(elm.Resistor().right().tox(fb_x).label(f'$R_3$\n{r3_kohm:.2f} kΩ', loc='top', fontsize=8))
        d.add(elm.Line().down().toy(fb_y))

        # Series Capacitor C1 going left from sum_node to Multiple-Feedback Junction J
        d.add(elm.Capacitor().left().at((sx, sy)).length(1.65).label(f'$C_1$\n{c_uf:.3f} µF', loc='top', fontsize=8))
        junc_node = d.add(elm.Dot())
        jx, jy = junc_node.center

        # Shunt Resistor R2 from Junction J down to Ground
        d.add(elm.Resistor().down().at((jx, jy)).length(1.55).label(f'$R_2$\n{r2_kohm:.2f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Series Input Resistor R1 from Junction J left to AC Source vin
        d.add(elm.Resistor().left().at((jx, jy)).length(1.65).label(f'$R_1$\n{r1_kohm:.2f} kΩ', loc='top', fontsize=8))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, jy)).length(0.25))
        d.add(elm.SourceSin().down().length(1.25).label('$v_{in}$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # Outer Feedback Capacitor C2 from Junction J over R3 back to Output fb_tap
        c2_y = r3_y + 1.05
        d.add(elm.Line().up().at((jx, jy)).toy(c2_y))
        d.add(elm.Capacitor().right().tox(fb_x).label(f'$C_2$\n{c_uf:.3f} µF', loc='top', fontsize=8))
        d.add(elm.Line().down().toy(r3_y))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Exact Multiple-Feedback Narrow Band-Pass Transfer Function
# -------------------------------------------------------------------------
def analyze_narrow_bpf(R1, R2, R3, C, freqs):
    # Exact complex transfer function H(s) = Vo / Vin for C1 = C2 = C:
    # H(s) = -(s / (R1*C)) / (s^2 + s * (2 / (R3*C)) + (R1 + R2) / (R1*R2*R3*C^2))
    wo = np.sqrt((R1 + R2) / (R1 * R2 * R3 * (C ** 2)))
    fc = wo / (2.0 * np.pi)
    bw = 1.0 / (np.pi * R3 * C)
    q_factor = fc / bw
    af_mag = R3 / (2.0 * R1)

    fl = np.sqrt(fc ** 2 + (bw / 2.0) ** 2) - (bw / 2.0)
    fh = np.sqrt(fc ** 2 + (bw / 2.0) ** 2) + (bw / 2.0)

    s = 1j * 2.0 * np.pi * freqs
    H_jw = -(s / (R1 * C)) / ((s ** 2) + s * (2.0 / (R3 * C)) + (wo ** 2))
    gain_mag = np.abs(H_jw)
    gain_db = 20.0 * np.log10(np.maximum(gain_mag, 1e-12))
    phase_deg = np.degrees(np.angle(H_jw))

    return fc, fl, fh, bw, q_factor, af_mag, gain_mag, gain_db, phase_deg

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Frequency Plot)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Design Mode & Parameters")
    design_mode = st.radio(
        "Parameter Input Mode",
        ["Design from Specs (fC, Q, AF, C)", "Direct Component Values (R1, R2, R3, C)"],
        index=0,
        help="Gayakwad Section 7.8 computes R1, R2, R3 from target fC, Q, AF, and C."
    )

    if design_mode == "Design from Specs (fC, Q, AF, C)":
        fc_target = st.slider(
            "Center Frequency $f_C$ (Hz)",
            min_value=200,
            max_value=5000,
            value=1000,
            step=50,
            help="Gayakwad Example 7-7 default: fC = 1.0 kHz."
        )
        q_target = st.slider(
            "Quality Factor $Q$ ($Q < 10$)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.2,
            help="Narrow band-pass filters typically use Q between 1 and 10 (default Q = 3)."
        )
        max_af_allowed = max(0.5, round(2.0 * (q_target ** 2) - 0.5, 1))
        af_target = st.slider(
            "Center-Frequency Gain $A_F$",
            min_value=0.5,
            max_value=min(25.0, max_af_allowed),
            value=min(10.0, max_af_allowed),
            step=0.5,
            help="Must satisfy AF < 2·Q² so R2 > 0 (default AF = 10 for Q = 3)."
        )
        c_uf = st.slider(
            "Capacitors $C_1 = C_2 = C$ (µF)",
            min_value=0.001,
            max_value=0.100,
            value=0.010,
            step=0.001,
            format="%.3f",
            help="Gayakwad Example 7-7 default: C1 = C2 = 0.01 µF."
        )

        C = c_uf * 1e-6
        R1 = q_target / (2.0 * np.pi * fc_target * C * af_target)
        R2 = q_target / (2.0 * np.pi * fc_target * C * (2.0 * (q_target ** 2) - af_target))
        R3 = q_target / (np.pi * fc_target * C)
        r1_kohm, r2_kohm, r3_kohm = R1 / 1e3, R2 / 1e3, R3 / 1e3
    else:
        r1_kohm = st.slider("Input Resistor $R_1$ (kΩ)", min_value=0.5, max_value=50.0, value=4.77, step=0.05)
        r2_kohm = st.slider("Shunt Resistor $R_2$ (kΩ)", min_value=0.5, max_value=50.0, value=5.97, step=0.05)
        r3_kohm = st.slider("Feedback Resistor $R_3$ (kΩ)", min_value=5.0, max_value=300.0, value=95.49, step=0.5)
        c_uf = st.slider("Capacitors $C_1 = C_2 = C$ (µF)", min_value=0.001, max_value=0.100, value=0.010, step=0.001, format="%.3f")

        R1 = r1_kohm * 1e3
        R2 = r2_kohm * 1e3
        R3 = r3_kohm * 1e3
        C = c_uf * 1e-6

    st.subheader("2. Load, Test Signal & Supply")
    rl_kohm = st.slider("Load Resistor $R_L$ (kΩ)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    vin_pp = st.slider("Input Signal $V_{in(pp)}$ (V)", min_value=0.2, max_value=3.0, value=1.0, step=0.2)
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)

# -------------------------------------------------------------------------
# Primary Filter Analysis
# -------------------------------------------------------------------------
vsat = max(vcc - 1.0, 0.5)
wo_init = np.sqrt((R1 + R2) / (R1 * R2 * R3 * (C ** 2)))
fc_init = wo_init / (2.0 * np.pi)

freqs = np.logspace(np.log10(max(fc_init / 15.0, 10.0)), np.log10(fc_init * 15.0), 800)
fc, fl, fh, bw, q_factor, af_mag, gain_mag, gain_db, phase_deg = analyze_narrow_bpf(R1, R2, R3, C, freqs)
af_db = 20.0 * np.log10(af_mag)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, Frequency Response & Tabs
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Center Frequency ($f_C$)", format_eng(fc, "Hz"))
    m2.metric("Bandwidth ($BW = f_H - f_L$)", f"{format_eng(bw, 'Hz')} (Q = {q_factor:.2f})")
    m3.metric("Center Gain ($A_F$)", f"{af_mag:.2f} ({af_db:+.2f} dB)")
    m4.metric("Cutoff Freqs ($f_L, f_H$)", f"{fl:.0f} Hz, {fh:.0f} Hz")

    st.success(
        f"**Narrow Band-Pass Filter ($Q = {q_factor:.2f}$):** Center frequency **$f_C = {fc:.2f}\\text{{ Hz}}$**, "
        f"$-3\\text{{ dB}}$ cutoff frequencies **$f_L = {fl:.1f}\\text{{ Hz}}$** and **$f_H = {fh:.1f}\\text{{ Hz}}$**, "
        f"bandwidth **$BW = {bw:.2f}\\text{{ Hz}}$**, and center gain **$A_F = {af_mag:.2f}$** (**${af_db:+.2f}\\text{{ dB}}$**)."
    )

    # Top row: Circuit Diagram (Left) + Frequency Response Curve (Right)
    schem_col, resp_col = st.columns([1.12, 1.38], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-13a*)**")
        fig_sch = draw_narrow_bpf_schematic(r1_kohm, r2_kohm, r3_kohm, c_uf, rl_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Component Values:** $R_1 = {r1_kohm:.2f}\\text{{ k}}\\Omega$, $R_2 = {r2_kohm:.2f}\\text{{ k}}\\Omega$, "
            f"$R_3 = {r3_kohm:.2f}\\text{{ k}}\\Omega$, $C_1 = C_2 = {c_uf:.3f}\\text{{ }}\\mu\\text{{F}}$."
        )

    with resp_col:
        st.markdown("**Frequency Response Curve (*Gayakwad Fig. 7-13b*)**")
        fig_freq, (ax_lin, ax_db) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        # Top Plot: Linear Voltage Gain Curve (matching Gayakwad Fig. 7-13b)
        ax_lin.semilogx(freqs, gain_mag, color="#1f77b4", linewidth=2.3, label="Voltage Gain $|v_o/v_{in}|$")
        ax_lin.axhline(af_mag, color="#555555", linestyle=":", linewidth=1.1, label=f"$A_F = {af_mag:.2f}$")
        ax_lin.axhline(0.7071 * af_mag, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$0.707 A_F = {0.7071*af_mag:.2f}$")
        ax_lin.axvline(fc, color="#d62728", linestyle="--", linewidth=1.1, label=f"$f_C = {fc:.0f}$ Hz")
        ax_lin.axvline(fl, color="#2ca02c", linestyle=":", linewidth=1.1)
        ax_lin.axvline(fh, color="#2ca02c", linestyle=":", linewidth=1.1)
        ax_lin.annotate(
            "", xy=(fh, 0.7071 * af_mag), xytext=(fl, 0.7071 * af_mag),
            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.8)
        )
        ax_lin.text(fc, 0.7071 * af_mag * 0.78, f"BW = {bw:.1f} Hz", ha="center", fontsize=8.5, fontweight="bold", color="#2ca02c")
        ax_lin.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_lin.set_ylabel("Gain $|v_o / v_{in}|$", fontsize=9, fontweight="bold")
        ax_lin.set_title(f"Linear Band-Pass Response ($f_L = {fl:.0f}$ Hz, $f_C = {fc:.0f}$ Hz, $f_H = {fh:.0f}$ Hz)", fontsize=9.5, fontweight="bold")
        ax_lin.legend(loc="upper right", fontsize=8)

        # Bottom Plot: Logarithmic Gain Response (dB)
        ax_db.semilogx(freqs, gain_db, color="#d62728", linewidth=2.2, label="Gain $20\\log_{10}|v_o/v_{in}|$ (dB)")
        ax_db.axhline(af_db - 3.01, color="#2ca02c", linestyle="--", linewidth=1.1, label=f"$-3$ dB ({af_db - 3.01:.2f} dB)")
        ax_db.axvline(fc, color="#d62728", linestyle="--", linewidth=1.1)
        ax_db.scatter([fl, fc, fh], [af_db - 3.01, af_db, af_db - 3.01], color="#1f77b4", zorder=5, s=30)
        ax_db.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_db.set_xlabel("Frequency $f$ (Hz) [Log Scale]", fontsize=9, fontweight="bold")
        ax_db.set_ylabel("Gain (dB)", fontsize=9, fontweight="bold")
        ax_db.set_title("Logarithmic Gain Response (dB vs. Frequency)", fontsize=9.5, fontweight="bold")
        ax_db.legend(loc="lower center", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_freq)
        plt.close(fig_freq)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Repeat for Another Center Frequency (Same Gain) + Obs + Theory
    # ---------------------------------------------------------------------
    tab_repeat, tab_obs, tab_theory = st.tabs([
        "🔁 Repeat for Another Center Frequency (Same Gain AF)",
        "📊 Frequency Response Observation Table",
        "📖 Theory & Design Equations (Gayakwad)"
    ])

    with tab_repeat:
        st.markdown(
            f"Per **Gayakwad Eq. (7-19)**, changing only resistor **$R_2$** to **$R_2' = R_2 (f_C / f_C')^2$** shifts the "
            f"center frequency from **$f_C = {fc:.1f}\\text{{ Hz}}$** to a new center frequency **$f_C'$** while keeping "
            f"**$R_1 = {r1_kohm:.2f}\\text{{ k}}\\Omega$**, **$R_3 = {r3_kohm:.2f}\\text{{ k}}\\Omega$**, "
            f"the center-frequency gain **$A_F = {af_mag:.2f}$**, and the bandwidth **$BW = {bw:.1f}\\text{{ Hz}}$** constant:"
        )

        fc_new_target = st.slider(
            "Select Second Center Frequency $f_C'$ (Hz) [Same Gain $A_F$]",
            min_value=300,
            max_value=4000,
            value=2000 if abs(fc - 2000) > 200 else 1500,
            step=50,
            help="Gayakwad Example 7-8 changes center frequency from 1 kHz to 1.5 kHz by adjusting R2 alone."
        )

        # Gayakwad Eq. (7-19): R2' = R2 * (fC / fC')^2
        R2_new = R2 * ((fc / fc_new_target) ** 2)
        f_min_cmp = min(fc, fc_new_target) / 10.0
        f_max_cmp = max(fc, fc_new_target) * 10.0
        freqs_cmp = np.logspace(np.log10(max(f_min_cmp, 10.0)), np.log10(f_max_cmp), 900)

        fc_1, fl_1, fh_1, bw_1, q_1, af_1, g_1, gdb_1, _ = analyze_narrow_bpf(R1, R2, R3, C, freqs_cmp)
        fc_2, fl_2, fh_2, bw_2, q_2, af_2, g_2, gdb_2, _ = analyze_narrow_bpf(R1, R2_new, R3, C, freqs_cmp)

        fig_rep, ax_rep = plt.subplots(figsize=(8.8, 3.9))
        ax_rep.semilogx(freqs_cmp, g_1, color="#1f77b4", linewidth=2.3,
                        label=f"Trial 1: $f_C = {fc_1:.1f}$ Hz ($R_2 = {R2/1e3:.2f}$ kΩ, $A_F = {af_1:.2f}$, $BW = {bw_1:.1f}$ Hz)")
        ax_rep.semilogx(freqs_cmp, g_2, color="#d62728", linewidth=2.3, linestyle="--",
                        label=f"Trial 2: $f_C' = {fc_2:.1f}$ Hz ($R_2' = {R2_new/1e3:.2f}$ kΩ, $A_F' = {af_2:.2f}$, $BW' = {bw_2:.1f}$ Hz)")
        ax_rep.axhline(af_1, color="#555555", linestyle=":", linewidth=1.1, label=f"Constant Peak Gain $A_F = {af_1:.2f}$")
        ax_rep.axhline(0.7071 * af_1, color="#2ca02c", linestyle=":", linewidth=1.1, label=f"$-3$ dB Level ($0.707 A_F = {0.7071*af_1:.2f}$)")
        ax_rep.axvline(fc_1, color="#1f77b4", linestyle=":", linewidth=1.0)
        ax_rep.axvline(fc_2, color="#d62728", linestyle=":", linewidth=1.0)
        ax_rep.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_rep.set_xlabel("Frequency $f$ (Hz) [Log Scale]", fontsize=9, fontweight="bold")
        ax_rep.set_ylabel("Voltage Gain $|v_o / v_{in}|$", fontsize=9, fontweight="bold")
        ax_rep.set_title("Comparison of Two Center Frequencies ($f_C$ vs. $f_C'$) at Identical Passband Gain $A_F$", fontsize=9.5, fontweight="bold")
        ax_rep.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_rep)
        plt.close(fig_rep)

        st.markdown("**Comparison Table (Two Center Frequencies at Same Gain $A_F$)**")
        st.table([
            {
                "Trial": "Trial 1 (Initial fC)",
                "R1 (kΩ)": f"{R1/1e3:.2f}",
                "R2 (kΩ)": f"{R2/1e3:.2f}",
                "R3 (kΩ)": f"{R3/1e3:.2f}",
                "Center Freq fC (Hz)": f"{fc_1:.2f}",
                "Cutoff Freqs fL, fH (Hz)": f"{fl_1:.1f}, {fh_1:.1f}",
                "Bandwidth BW (Hz)": f"{bw_1:.2f}",
                "Quality Factor Q": f"{q_1:.2f}",
                "Gain AF": f"{af_1:.2f}"
            },
            {
                "Trial": "Trial 2 (Shifted fC')",
                "R1 (kΩ)": f"{R1/1e3:.2f}",
                "R2 (kΩ)": f"{R2_new/1e3:.2f}",
                "R3 (kΩ)": f"{R3/1e3:.2f}",
                "Center Freq fC (Hz)": f"{fc_2:.2f}",
                "Cutoff Freqs fL, fH (Hz)": f"{fl_2:.1f}, {fh_2:.1f}",
                "Bandwidth BW (Hz)": f"{bw_2:.2f}",
                "Quality Factor Q": f"{q_2:.2f}",
                "Gain AF": f"{af_2:.2f}"
            }
        ])

    with tab_obs:
        st.markdown(
            f"Tabulated gain and phase response readings across frequencies around **$f_C = {fc:.2f}\\text{{ Hz}}$** "
            f"with constant input **$V_{{in(pp)}} = {vin_pp:.2f}\\text{{ V}}$**:"
        )
        test_freqs = np.array([
            0.2 * fc, 0.5 * fc, 0.7 * fc, fl, 0.95 * fc,
            fc, 1.05 * fc, fh, 1.4 * fc, 2.0 * fc, 5.0 * fc
        ])
        _, _, _, _, _, _, g_obs, gdb_obs, ph_obs = analyze_narrow_bpf(R1, R2, R3, C, test_freqs)
        obs_rows = []
        for idx, f_val in enumerate(test_freqs):
            vo_pp_val = min(g_obs[idx] * vin_pp, 2.0 * vsat)
            if np.isclose(f_val, fc):
                marker = "Center Frequency (fC)"
            elif np.isclose(f_val, fl):
                marker = "Lower -3 dB Cutoff (fL)"
            elif np.isclose(f_val, fh):
                marker = "Upper -3 dB Cutoff (fH)"
            elif fl < f_val < fh:
                marker = "Passband"
            else:
                marker = "Stopband"
            obs_rows.append({
                "Frequency f (Hz)": f"{f_val:.1f}",
                "Input Vin(pp) (V)": f"{vin_pp:.2f}",
                "Output Vo(pp) (V)": f"{vo_pp_val:.2f}",
                "Voltage Gain |Vo/Vin|": f"{g_obs[idx]:.3f}",
                "Gain (dB)": f"{gdb_obs[idx]:+.2f}",
                "Phase Shift (deg)": f"{ph_obs[idx]:+.1f}°",
                "Operating Point": marker
            })
        st.table(obs_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7.8, Figure 7-13*)

            1. **Multiple-Feedback Narrow Band-Pass Filter ($Q < 10$):**
               Unlike a wide band-pass filter formed by cascading separate high-pass and low-pass stages, a **narrow band-pass filter** uses a single op-amp with multiple feedback paths ($R_3$ and $C_2$) to achieve higher selectivity ($Q$ up to $10$)[cite: 9].

            2. **Design Equations (*Gayakwad Eqs. 7-16 to 7-18*, with $C_1 = C_2 = C$):**
               Given center frequency $f_C$, quality factor $Q$, center-frequency gain $A_F$ (where $A_F < 2 Q^2$), and capacitor $C$:
               $$R_1 = \\frac{{Q}}{{2\\pi f_C C A_F}} = {R1/1e3:.2f}\\text{{ k}}\\Omega$$
               $$R_2 = \\frac{{Q}}{{2\\pi f_C C (2 Q^2 - A_F)}} = {R2/1e3:.2f}\\text{{ k}}\\Omega$$
               $$R_3 = \\frac{{Q}}{{\\pi f_C C}} = {R3/1e3:.2f}\\text{{ k}}\\Omega$$

            3. **Analysis Equations (*Gayakwad Eqs. 7-14 & 7-15*):**
               * **Center Frequency:** $f_C = \\frac{{1}}{{2\\pi C}}\\sqrt{{\\frac{{R_1 + R_2}}{{R_1 R_2 R_3}}}} = {fc:.2f}\\text{{ Hz}}$
               * **Center-Frequency Gain:** $A_F = \\frac{{R_3}}{{2 R_1}} = {af_mag:.2f}$ ($180^\\circ$ phase inversion at $f_C$)
               * **Bandwidth & Quality Factor:** $BW = f_H - f_L = \\frac{{f_C}}{{Q}} = \\frac{{1}}{{\\pi R_3 C}} = {bw:.2f}\\text{{ Hz}}$

            4. **Changing Center Frequency Without Changing Gain (*Gayakwad Eq. 7-19*):**
               Because $A_F = \\frac{{R_3}}{{2 R_1}}$ and $BW = \\frac{{1}}{{\\pi R_3 C}}$ do not depend on $R_2$, the center frequency can be tuned from $f_C$ to a new value $f_C'$ at the exact same gain and bandwidth simply by changing $R_2$ to:
               $$R_2' = R_2 \\left(\\frac{{f_C}}{{f_C'}}\\right)^2$$
            """
        )