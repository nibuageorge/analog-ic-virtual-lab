import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 8: RC Phase Shift Oscillator", layout="wide")
render_sidebar_header()

st.title("Experiment 8: RC Phase Shift Oscillator using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct an **RC Phase Shift Oscillator** using **$\mu$A741**"
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 7-18 RC Phase Shift Oscillator Schematic
# -------------------------------------------------------------------------
def draw_rc_phase_shift_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, rom_kohm, use_zener, vcc):
    rf_label = f"{rf_kohm/1000:.2f} MΩ" if rf_kohm >= 1000 else f"{rf_kohm:.0f} kΩ"

    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.4)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Offset Minimizing Resistor R_OM on Non-Inverting (+) Pin (kept compact above RC ladder)
        noninv_x, noninv_y = op.in2
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.45))
        d.add(elm.Resistor().down().length(0.95).label(f'$R_{{OM}}$\n{rom_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # --- Output Node & Back-to-Back Zener Clamp D1, D2 ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        z_tap = d.add(elm.Dot())
        zx, zy = z_tap.center

        d.add(elm.Line().right().at((zx, zy)).length(0.75).label('$v_o$', loc='right', fontsize=9.5))

        # Back-to-Back Zeners D1 & D2 to Ground (labels cleanly on right side)
        d.add(elm.Line().down().at((zx, zy)).length(0.25))
        if use_zener:
            d.add(elm.Zener().down().length(0.95).label('$D_1$', loc='bottom', fontsize=8.5))
            d.add(elm.Zener().down().reverse().length(0.95).label('$D_2$', loc='bottom', fontsize=8.5))
        else:
            d.add(elm.Line().down().length(1.9).linestyle('--'))
        d.add(elm.Ground())

        # --- Upper Inverting Amplifier Network (R1 & Potentiometer RF) ---
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.25
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.5))
        d.add(elm.Line().up().toy(top_y))
        sum_node = d.add(elm.Dot())
        sx, sy = sum_node.center

        # Feedback Potentiometer RF connected as rheostat to Output fb_tap
        d.add(elm.Potentiometer().right().at((sx, sy)).tox(fb_x).label(f'$R_F$\n({rf_label})', loc='top', fontsize=8))
        d.add(elm.Line().down().at((fb_x, sy)).toy(fb_y))

        # Input Resistor R1 going left from sum_node
        d.add(elm.Resistor().left().at((sx, sy)).length(2.15).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        left_bus_x = d.here[0]

        # --- Lower 3-Stage RC Phase-Shift Ladder Network (dropped well below R_OM) ---
        ladder_y = noninv_y - 2.45
        d.add(elm.Line().down().at((left_bus_x, sy)).toy(ladder_y))
        n1 = d.add(elm.Dot())
        n1_x, n1_y = n1.center

        # Shunt Resistor R #1 to Ground (loc='bottom' puts label cleanly beside vertical resistor)
        d.add(elm.Resistor().down().at((n1_x, n1_y)).length(1.3).label(f'$R$\n{r_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Series Capacitor C #1
        d.add(elm.Capacitor().right().at((n1_x, n1_y)).length(1.75).label(f'$C$\n{c_uf:.2f} µF', loc='top', fontsize=8))
        n2 = d.add(elm.Dot())
        n2_x, n2_y = n2.center

        # Shunt Resistor R #2 to Ground
        d.add(elm.Resistor().down().at((n2_x, n2_y)).length(1.3).label(f'$R$\n{r_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Series Capacitor C #2
        d.add(elm.Capacitor().right().at((n2_x, n2_y)).length(1.75).label(f'$C$\n{c_uf:.2f} µF', loc='top', fontsize=8))
        n3 = d.add(elm.Dot())
        n3_x, n3_y = n3.center

        # Shunt Resistor R #3 to Ground
        d.add(elm.Resistor().down().at((n3_x, n3_y)).length(1.3).label(f'$R$\n{r_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        # Series Capacitor C #3 back to Output fb_tap
        d.add(elm.Capacitor().right().at((n3_x, n3_y)).tox(fb_x).label(f'$C$\n{c_uf:.2f} µF', loc='top', fontsize=8))
        d.add(elm.Line().up().toy(fb_y))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Time-Domain Simulation of 3-Stage RC Phase Shift Oscillator
# -------------------------------------------------------------------------
def simulate_rc_phase_shift(R, C, R1, Rf, use_zener, vz, vcc, num_cycles=5, num_pts=3500):
    vsat = max(vcc - 1.0, 0.5)
    # Gayakwad Eq. (7-22a): fo = 1 / (2*pi*sqrt(6)*R*C) = 0.065 / (R*C)
    fo = 0.065 / (R * C)
    T = 1.0 / fo
    wo = 2.0 * np.pi * fo

    # Inverting amplifier voltage gain magnitude: |Av| = Rf / R1 (Req >= 29)
    av_mag = Rf / R1

    # Clamped output peak voltage: ±(Vz + 0.7V) with back-to-back Zeners, else ±Vsat
    v_clamp = min(vz + 0.7, vsat) if use_zener else vsat

    t_disp = np.linspace(0.0, num_cycles * T, num_pts)

    if av_mag < 28.8:
        # Below Barkhausen threshold (Rf < 29*R1): oscillations decay to 0 V
        decay_rate = (29.0 - av_mag) * 0.04 * wo
        v_out_disp = 0.8 * np.exp(-decay_rate * t_disp) * np.sin(wo * t_disp)
        v_fb_disp = -(v_out_disp / 29.0)
        thd_pct = 0.0
        return t_disp, v_fb_disp, v_out_disp, fo, T, av_mag, v_clamp, thd_pct, vsat

    # Sustained oscillation (|Av| >= 29): 180° phase shift between output vo and RC feedback v_fb
    overdrive = 1.0 + 0.08 * max(av_mag - 29.0, 0.0)
    raw_wave = overdrive * v_clamp * np.sin(wo * t_disp)

    if use_zener:
        # Smooth Zener diode knee clamping at ±(Vz + 0.7V)
        v_out_disp = v_clamp * np.tanh(raw_wave / (0.92 * v_clamp))
    else:
        v_out_disp = np.clip(raw_wave, -vsat, vsat)

    # Feedback signal after 3-stage RC network is inverted (180° out of phase) and attenuated by 1/29
    v_fb_disp = -(v_out_disp / 29.0)

    # Compute THD from FFT
    v_ac = v_out_disp - np.mean(v_out_disp)
    fft_vals = np.abs(np.fft.rfft(v_ac * np.hanning(len(v_ac))))
    fund_idx = np.argmax(fft_vals[1:]) + 1
    fund_amp = fft_vals[fund_idx]
    harmonics_sq = np.sum(fft_vals[2 * fund_idx::fund_idx] ** 2)
    thd_pct = (np.sqrt(harmonics_sq) / fund_amp) * 100.0 if fund_amp > 1e-6 else 0.0

    return t_disp, v_fb_disp, v_out_disp, fo, T, av_mag, v_clamp, thd_pct, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. 3-Stage RC Ladder Network ($R, C$)")
    r_kohm = st.slider(
        "Ladder Resistor $R$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=3.3,
        step=0.1,
        help="Gayakwad Fig. 7-18 default: R = 3.3 kΩ. Sets fo = 0.065 / (R·C)."
    )
    c_uf = st.slider(
        "Ladder Capacitor $C$ (µF)",
        min_value=0.005,
        max_value=0.500,
        value=0.100,
        step=0.005,
        format="%.3f",
        help="Gayakwad Fig. 7-18 default: C = 0.1 µF (gives fo ≈ 197 Hz, T ≈ 5 ms)."
    )

    st.subheader("2. Amplifier Gain & Zener Clamp ($R_1, R_F, D_1/D_2$)")
    r1_kohm = st.slider(
        "Input Resistor $R_1$ (kΩ)",
        min_value=5.0,
        max_value=100.0,
        value=33.0,
        step=1.0,
        help="Gayakwad recommends R1 >= 10·R (default 33 kΩ) to prevent loading the RC network."
    )
    rf_kohm = st.slider(
        "Feedback Potentiometer $R_F$ (kΩ)",
        min_value=100.0,
        max_value=1500.0,
        value=1000.0,
        step=10.0,
        help="Must satisfy RF >= 29·R1 (for R1 = 33 kΩ, 29·R1 = 957 kΩ; default 1 MΩ pot)."
    )
    use_zener = st.checkbox(
        "Enable Back-to-Back Zener Clamp ($D_1, D_2$: 1N4735)",
        value=True,
        help="Clamps output swing symmetrically at ±(Vz + 0.7V) = ±6.9 V."
    )
    vz = st.slider(
        "Zener Breakdown Voltage $V_Z$ (V)",
        min_value=3.3,
        max_value=12.0,
        value=6.2,
        step=0.1,
        disabled=not use_zener,
        help="1N4735 Zener voltage is 6.2 V -> ±(6.2 + 0.7) = ±6.9 V peak output."
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=3, max_value=8, value=5, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R = r_kohm * 1e3
C = c_uf * 1e-6
R1 = r1_kohm * 1e3
Rf = rf_kohm * 1e3
rom_kohm = r1_kohm

t_disp, v_fb_disp, v_out_disp, fo, T, av_mag, v_clamp, thd_pct, vsat = simulate_rc_phase_shift(
    R, C, R1, Rf, use_zener, vz, vcc, num_cycles=num_cycles
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Inverting Gain ($|A_v| = R_F/R_1$)", f"{av_mag:.1f} (Req ≥ 29.0)")
    m3.metric("Output Peak ($\\pm V_p$)", f"±{np.max(np.abs(v_out_disp)):.2f} V")
    m4.metric("Time Period ($T$)", format_eng(T, "s"))

    if av_mag < 28.8:
        st.error(
            f"**Oscillation Damps Out ($|A_v| = R_F/R_1 = {av_mag:.1f} < 29.0$):** "
            f"Gayakwad Eq. (7-22b) requires **$R_F \\ge 29 R_1 = {29*r1_kohm:.0f}\\text{{ k}}\\Omega$** so that $|A_v| \\beta \\ge 1$. "
            f"Increase potentiometer $R_F$ above **{29*r1_kohm:.0f} kΩ**."
        )
    elif r1_kohm < 10.0 * r_kohm:
        st.warning(
            f"**Loading Effect Notice ($R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega < 10 R = {10*r_kohm:.1f}\\text{{ k}}\\Omega$):** "
            f"Gayakwad recommends **$R_1 \\ge 10 R$** so the inverting amplifier does not load the third $RC$ section and shift $f_o$."
        )
    else:
        st.success(
            f"**Sustained Sinusoidal Oscillation ($f_o = {fo:.2f}\\text{{ Hz}}$, $T = {T*1000:.2f}\\text{{ ms}}$):** "
            f"Gain $|A_v| = {av_mag:.1f} \\ge 29$ overcomes the $1/29$ ladder attenuation, and Zeners clamp the peak output at $\\pm {v_clamp:.2f}\\text{{ V}}$."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-18*)**")
        fig_sch = draw_rc_phase_shift_schematic(r_kohm, c_uf, r1_kohm, rf_kohm, rom_kohm, use_zener, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Design Check:** $R_1 = {r1_kohm:.0f}\\text{{ k}}\\Omega \\ge 10 R$ ($33\\text{{ k}}\\Omega$), "
            f"$R_{{OM}} = {rom_kohm:.0f}\\text{{ k}}\\Omega$, $f_o = \\frac{{0.065}}{{R C}} = {fo:.1f}\\text{{ Hz}}$."
        )

    with cro_col:
        st.markdown("**RC Ladder Feedback ($180^\\circ$) & Output ($v_o$) (CRO)**")
        fig_cro, (ax_fb, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: 180° phase-shifted feedback signal from 3-stage RC ladder
        ax_fb.plot(t_ms, v_fb_disp * 1000.0, color="#1f77b4", linewidth=2.0, label="RC Network Feedback $v_{fb}(t)$ ($180^\\circ$ shifted)")
        y_lim_fb = max(np.max(np.abs(v_fb_disp * 1000.0)) * 1.35, 50.0)
        ax_fb.set_ylim(-y_lim_fb, y_lim_fb)
        style_cro_axes(
            ax_fb,
            title=f"CH1: 3-Stage RC Ladder Output $v_{{fb}}(t)$ ($180^\\circ$ phase shift, $\\beta = 1/29$)",
            xlabel="",
            ylabel="$v_{fb}$ (mV)"
        )
        ax_fb.legend(loc="upper right", fontsize=8)

        # Channel 2: Clamped Sine Wave Output vo(t)
        ax_out.plot(t_ms, v_out_disp, color="#d62728", linewidth=2.2, label="Oscillator Output $v_o(t)$")
        if use_zener:
            ax_out.axhline(v_clamp, color="#2ca02c", linestyle="--", linewidth=1, label=f"$+(V_Z + V_{{D1}}) = +{v_clamp:.1f}$ V")
            ax_out.axhline(-v_clamp, color="#ff7f0e", linestyle="--", linewidth=1, label=f"$-(V_Z + V_{{D2}}) = -{v_clamp:.1f}$ V")
        ax_out.set_ylim(-max(v_clamp, vsat * 0.7) * 1.3, max(v_clamp, vsat * 0.7) * 1.3)
        style_cro_axes(
            ax_out,
            title=f"CH2: Phase Shift Oscillator Output $v_o(t)$ ($f_o = {fo:.1f}$ Hz, $T = {T*1000:.2f}$ ms)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Multi-Frequency Comparison (3 Frequencies) + Theory
    # ---------------------------------------------------------------------
    tab_multi, tab_theory = st.tabs([
        "🔁 Repeat for Three Frequencies ($f_1, f_2, f_3$)",
        "📖 Theory & Design Procedure (Gayakwad)"
    ])

    with tab_multi:
        st.markdown(
            "Select three RC ladder resistor values ($R_a, R_b, R_c$) with fixed $C = "
            f"{c_uf:.3f}\\text{{ µF}}$, $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$, and $R_F = {rf_kohm:.0f}\\text{{ k}}\\Omega$ "
            "to compare the sine-wave outputs across **three different frequencies**:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            ra_kohm = st.number_input("Trial 1: Resistor $R_a$ (kΩ)", min_value=1.0, max_value=50.0, value=1.5, step=0.1)
        with fcol2:
            rb_kohm = st.number_input("Trial 2: Resistor $R_b$ (kΩ)", min_value=1.0, max_value=50.0, value=3.3, step=0.1)
        with fcol3:
            rc_kohm = st.number_input("Trial 3: Resistor $R_c$ (kΩ)", min_value=1.0, max_value=50.0, value=6.8, step=0.1)

        trials = [
            ("Trial 1 ($f_1$)", ra_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", rb_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", rc_kohm * 1e3, "#d62728"),
        ]

        max_R = max(ra_kohm, rb_kohm, rc_kohm) * 1e3
        T_max_trial = (max_R * C) / 0.065
        t_common_span = 4.0 * T_max_trial

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r_val, color) in enumerate(trials):
            fo_i = 0.065 / (r_val * C)
            T_i = 1.0 / fo_i
            cycles_needed = max(4, int(np.ceil(t_common_span / T_i)))
            t_i, _, vout_i, _, _, _, v_cl_i, thd_i, _ = simulate_rc_phase_shift(
                r_val, C, R1, Rf, use_zener, vz, vcc, num_cycles=cycles_needed, num_pts=4000
            )
            mask_i = t_i <= t_common_span
            ax = axes_cmp[idx]
            ax.plot(t_i[mask_i] * 1000.0, vout_i[mask_i], color=color, linewidth=2.0)
            ax.set_ylim(-v_cl_i * 1.25, v_cl_i * 1.25)
            style_cro_axes(
                ax,
                title=f"{label}: $R = {r_val/1e3:.2f}$ kΩ, $C = {c_uf:.3f}$ µF  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                xlabel="Time (ms)" if idx == 2 else "",
                ylabel="$v_o$ (V)"
            )
            table_rows.append({
                "Trial": label,
                "Ladder Resistor R (kΩ)": f"{r_val/1e3:.2f}",
                "Ladder Capacitor C (µF)": f"{c_uf:.3f}",
                "Time Period T (ms)": f"{T_i*1000:.3f}",
                "Frequency fo (Hz)": f"{fo_i:.2f}",
                "Output Vo(pp) (V)": f"{np.ptp(vout_i):.2f}",
                "THD (%)": f"{thd_i:.2f}%"
            })

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("**Observation Table (Three Frequencies)**")
        st.table(table_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7.9, Fig. 7-18*)

            1. **Principle of Operation ($360^\\circ$ Loop Phase Shift):**
               * The op-amp is configured as an **inverting amplifier**, which introduces a **$180^\\circ$ phase shift** between its input and output[cite: 8].
               * The three identical cascaded $RC$ sections provide an additional **$180^\\circ$ phase shift** ($60^\\circ$ per stage at resonance), making the total phase shift around the closed loop equal to **$360^\\circ$**[cite: 8].

            2. **Frequency of Oscillation (*Gayakwad Eq. 7-22a*):**
               $$f_o = \\frac{{1}}{{2\\pi \\sqrt{{6}} R C}} = \\frac{{0.065}}{{R C}} = {fo:.2f}\\text{{ Hz}}$$

            3. **Barkhausen Gain Condition (*Gayakwad Eq. 7-22b*):**
               At the oscillation frequency $f_o$, the attenuation of the three-stage $RC$ ladder network is $\\beta = \\frac{{1}}{{29}}$. Therefore, the inverting amplifier gain must be at least **$29$**:
               $$|A_v| = \\frac{{R_F}}{{R_1}} = 29 \\implies R_F = 29 R_1$$
               *(To prevent the inverting amplifier from loading the third $RC$ stage, select $R_1 \\ge 10 R$, e.g., $R = 3.3\\text{{ k}}\\Omega$ and $R_1 = 33\\text{{ k}}\\Omega$, with $R_F = 1\\text{{ M}}\\Omega$ potentiometer[cite: 8].)*

            4. **Output Amplitude Clamping ($D_1, D_2$):**
               Back-to-back Zener diodes (**1N4735**, $V_Z = 6.2\\text{{ V}}$) clamp the sinusoidal peaks symmetrically at $\\pm(V_Z + 0.7\\text{{ V}}) = \\pm 6.9\\text{{ V}}$[cite: 8].
            """
        )
