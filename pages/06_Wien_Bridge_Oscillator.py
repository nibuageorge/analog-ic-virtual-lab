import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 6: Wien Bridge Oscillator", layout="wide")
render_sidebar_header()

st.title("Experiment 6: Wien Bridge Oscillator with Amplitude Stabilization ($\mu$A741)")
st.markdown(
    "Design and construct a Wien bridge sine-wave oscillator with back-to-back Zener diode amplitude stabilization "
    "using **$\mu$A741**, study the output waveform and Barkhausen gain condition, and repeat for three frequencies "
    "(*Gayakwad, Section 7.4, Fig. 7-10*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Wien Bridge Oscillator Schematic (Fig. 7-10)
# -------------------------------------------------------------------------
def draw_wien_bridge_schematic(r_kohm, c_uf, ri_kohm, rf1_kohm, rf2_kohm, use_zener, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.5)

        # --- Op-Amp: Non-inverting (+) on top, Inverting (-) on bottom for clean bridge layout ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).flip().label('$\mu$A741', loc='center', fontsize=8.5))
        # Note: after .flip(), op.in2 (+) is on top, op.in1 (-) is on bottom
        d.add(elm.Line().up().at(op.vs).length(0.35).label(f'+{vcc:.0f}V (7)', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vd).length(0.35).label(f'-{vcc:.0f}V (4)', loc='bottom', fontsize=7.5))

        # Output Node
        d.add(elm.Line().right().at(op.out).length(0.8))
        out_node = d.add(elm.Dot())
        ox, oy = out_node.center
        d.add(elm.Line().right().at((ox, oy)).length(0.85).label('$v_o$ (6)', loc='right', fontsize=9))

        # =================================================================
        # UPPER BRANCH: Lead-Lag Frequency-Selective Network (to Pin 3, +)
        # =================================================================
        in_plus_x, in_plus_y = op.in2
        d.add(elm.Line().left().at((in_plus_x, in_plus_y)).length(1.1))
        pos_node = d.add(elm.Dot())
        px, py = pos_node.center

        # Parallel R2 || C2 to Ground from Non-Inverting Node
        d.add(elm.Resistor().down().at((px, py)).length(1.35).label(f'$R$\n{r_kohm:.1f} kΩ', loc='left', fontsize=8))
        d.add(elm.Ground())

        d.add(elm.Line().left().at((px, py)).length(0.95))
        d.add(elm.Capacitor().down().length(1.35).label(f'$C$\n{c_uf:.3f} µF', loc='left', fontsize=8))
        d.add(elm.Ground())

        # Series R1 + C1 from Non-Inverting Node over top to Output Node
        top_y = py + 1.35
        d.add(elm.Line().up().at((px, py)).toy(top_y))
        d.add(elm.Resistor().right().length(1.65).label(f'$R$\n{r_kohm:.1f} kΩ', loc='top', fontsize=8))
        d.add(elm.Capacitor().right().tox(ox).label(f'$C$\n{c_uf:.3f} µF', loc='top', fontsize=8))
        d.add(elm.Line().down().toy(oy))

        # =================================================================
        # LOWER BRANCH: Negative Feedback + Zener AGC (to Pin 2, -)
        # =================================================================
        in_minus_x, in_minus_y = op.in1
        d.add(elm.Line().left().at((in_minus_x, in_minus_y)).length(0.55))
        d.add(elm.Line().down().length(1.35))
        neg_node = d.add(elm.Dot())
        nx, ny = neg_node.center

        # Resistor R_i from Inverting Node to Ground
        d.add(elm.Resistor().down().at((nx, ny)).length(1.25).label(f'$R_i$\n{ri_kohm:.1f} kΩ', loc='left', fontsize=8))
        d.add(elm.Ground())

        # Feedback path: R_F1 in series with (R_F2 || D1-D2) to Output Node
        d.add(elm.Resistor().right().at((nx, ny)).length(1.45).label(f'$R_{{F1}}$\n{rf1_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        z_left = d.add(elm.Dot())
        zx1, zy1 = z_left.center

        d.add(elm.Resistor().right().at((zx1, zy1)).tox(ox).label(f'$R_{{F2}}$\n{rf2_kohm:.1f} kΩ', loc='bottom', fontsize=8))
        z_right = d.add(elm.Dot())
        zx2, zy2 = z_right.center
        d.add(elm.Line().up().at((zx2, zy2)).toy(oy))

        # Parallel Back-to-Back Zener Diodes D1 & D2 across R_F2
        if use_zener:
            z_bot_y = zy1 - 0.95
            d.add(elm.Line().down().at((zx1, zy1)).toy(z_bot_y))
            mid_zx = (zx1 + zx2) / 2.0
            d.add(elm.Zener().right().at((zx1, z_bot_y)).tox(mid_zx).label('$D_1$', loc='bottom', fontsize=7.5))
            d.add(elm.Zener().right().reverse().at((mid_zx, z_bot_y)).tox(zx2).label('$D_2$', loc='bottom', fontsize=7.5))
            d.add(elm.Line().up().at((zx2, z_bot_y)).toy(zy2))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Exact Nonlinear Time-Domain Simulation of Wien Bridge Oscillator
# -------------------------------------------------------------------------
def simulate_wien_oscillator(R, C, Ri, Rf1, Rf2, use_zener, vz, vcc, num_cycles=6, num_pts=3500):
    vsat = max(vcc - 1.5, 0.5)
    wo = 1.0 / (R * C)
    fo = wo / (2.0 * np.pi)

    # Small-signal non-inverting gain (when Zeners are OFF): A_v0 = 1 + (Rf1 + Rf2) / Ri
    av_small = 1.0 + (Rf1 + Rf2) / Ri
    # High-amplitude gain (when Zeners clamp Rf2): A_v_min = 1 + Rf1 / Ri
    av_clamped = 1.0 + Rf1 / Ri

    # State-space ODE for Wien lead-lag network:
    # Let x1 = v_plus (voltage across parallel RC), x2 = i_series * R
    # Normalized time tau = wo * t:
    # dx1/dtau = -2*x1 + v_o - x2
    # dx2/dtau = x1
    # where v_o = f_amp(x1)
    v_z_clamp = vz + 0.7  # Breakdown + forward diode drop across Rf2

    def amp_transfer(v_plus):
        # Small-signal linear output
        v_lin = av_small * v_plus
        if not use_zener:
            return np.clip(v_lin, -vsat, vsat)
        # Current through Ri is v_plus / Ri; voltage across Rf2 is (v_plus / Ri) * Rf2
        v_rf2 = v_plus * (Rf2 / Ri)
        if abs(v_rf2) <= v_z_clamp:
            v_out = v_lin
        else:
            # Zener clamps voltage across Rf2 to ±v_z_clamp
            v_rf2_clamped = np.sign(v_rf2) * v_z_clamp
            v_out = v_plus * (1.0 + Rf1 / Ri) + v_rf2_clamped
        return np.clip(v_out, -vsat, vsat)

    def wien_ode(tau, state):
        x1, x2 = state
        vo = amp_transfer(x1)
        dx1 = -2.0 * x1 + vo - x2
        dx2 = x1
        return [dx1, dx2]

    # Integrate for enough cycles to reach steady-state amplitude, then display last num_cycles
    settle_cycles = 28 if av_small > 3.0 else 10
    total_cycles = settle_cycles + num_cycles
    tau_end = 2.0 * np.pi * total_cycles
    tau_disp_start = 2.0 * np.pi * settle_cycles

    tau_eval = np.linspace(tau_disp_start, tau_end, num_pts)
    sol = solve_ivp(
        wien_ode,
        (0.0, tau_end),
        [0.02, 0.0],  # Tiny initial noise perturbation
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    t_disp = (sol.t - tau_disp_start) / wo
    v_plus_disp = sol.y[0]
    v_out_disp = np.array([amp_transfer(vp) for vp in v_plus_disp])

    # Estimate THD (Total Harmonic Distortion) from FFT of steady-state v_out
    v_ac = v_out_disp - np.mean(v_out_disp)
    fft_vals = np.abs(np.fft.rfft(v_ac * np.hanning(len(v_ac))))
    fund_idx = np.argmax(fft_vals[1:]) + 1
    fund_amp = fft_vals[fund_idx]
    if fund_amp > 1e-6:
        harmonics_sq = np.sum(fft_vals[2 * fund_idx::fund_idx] ** 2)
        thd_pct = (np.sqrt(harmonics_sq) / fund_amp) * 100.0
    else:
        thd_pct = 0.0

    return t_disp, v_plus_disp, v_out_disp, fo, 1.0 / fo, av_small, av_clamped, thd_pct, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Frequency-Selective Wien Network ($R, C$)")
    r_kohm = st.slider(
        "Wien Resistor $R$ ($R_1 = R_2 = R$, kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=15.9,
        step=0.1,
        help="Sets oscillation frequency fo = 1 / (2π R C). For R = 15.9 kΩ and C = 0.01 µF, fo ≈ 1.00 kHz."
    )
    c_uf = st.slider(
        "Wien Capacitor $C$ ($C_1 = C_2 = C$, µF)",
        min_value=0.001,
        max_value=0.200,
        value=0.010,
        step=0.001,
        format="%.3f"
    )

    st.subheader("2. Gain & Amplitude Stabilization ($R_i, R_{F1}, R_{F2}, D_1/D_2$)")
    ri_kohm = st.slider(
        "Input Ground Resistor $R_i$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )
    rf1_kohm = st.slider(
        "Fixed Feedback Resistor $R_{F1}$ (kΩ)",
        min_value=5.0,
        max_value=50.0,
        value=18.0,
        step=0.5,
        help="Choose RF1 < 2·Ri so clamped gain (1 + RF1/Ri) < 3 for clean sine-wave stabilization."
    )
    rf2_kohm = st.slider(
        "Zener-Shunted Resistor $R_{F2}$ (kΩ)",
        min_value=1.0,
        max_value=30.0,
        value=4.7,
        step=0.1,
        help="Choose RF1 + RF2 > 2·Ri so small-signal gain Av0 = 1 + (RF1+RF2)/Ri > 3 to start oscillations."
    )
    use_zener = st.checkbox(
        "Enable Back-to-Back Zener Stabilization ($D_1, D_2$)",
        value=True,
        help="Prevents op-amp saturation clipping and maintains low-distortion sinusoidal output."
    )
    vz = st.slider(
        "Zener Voltage $V_Z$ (V)",
        min_value=2.4,
        max_value=10.0,
        value=3.3,
        step=0.1,
        disabled=not use_zener
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=3, max_value=8, value=5, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R = r_kohm * 1e3
C = c_uf * 1e-6
Ri = ri_kohm * 1e3
Rf1 = rf1_kohm * 1e3
Rf2 = rf2_kohm * 1e3

t_disp, v_plus_disp, v_out_disp, fo, T, av_small, av_clamped, thd_pct, vsat = simulate_wien_oscillator(
    R, C, Ri, Rf1, Rf2, use_zener, vz, vcc, num_cycles=num_cycles
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Small-Signal Gain ($A_{v0}$)", f"{av_small:.2f} (Req > 3.00)")
    m3.metric("Output $V_{o(pp)}$", f"{np.ptp(v_out_disp):.2f} V")
    m4.metric("Waveform THD", f"{thd_pct:.2f}%")

    if av_small <= 3.0:
        st.error(
            f"**Oscillation Fails to Start ($A_{{v0}} = 1 + \\frac{{R_{{F1}} + R_{{F2}}}}{{R_i}} = {av_small:.2f} \\le 3.00$):** "
            f"Barkhausen criterion requires loop gain $A_v \\beta > 1$ ($A_{{v0}} > 3.00$ since $\\beta = 1/3$ at $f_o$). "
            f"Increase $R_{{F1}}$ or $R_{{F2}}$, or decrease $R_i$."
        )
    elif not use_zener:
        st.warning(
            f"**No Amplitude Stabilization ($D_1, D_2$ Disabled):** "
            f"Because $A_{{v0}} = {av_small:.2f} > 3$, oscillations grow until clipped by the op-amp saturation rails $\\pm V_{{sat}} = \\pm {vsat:.1f}\\text{{ V}}$ (THD = {thd_pct:.1f}%)."
        )
    elif av_clamped >= 3.0:
        st.warning(
            f"**Insufficient Gain Reduction ($1 + R_{{F1}}/R_i = {av_clamped:.2f} \\ge 3.00$):** "
            f"Even with Zener diodes conducting across $R_{{F2}}$, the gain remains above $3$, causing op-amp saturation clipping. Reduce $R_{{F1}}$ below $2 R_i = {2*ri_kohm:.1f}\\text{{ k}}\\Omega$."
        )
    else:
        st.success(
            f"**Stabilized Sinusoidal Oscillation ($f_o = {fo:.2f}\\text{{ Hz}}$, $V_{{o(pp)}} = {np.ptp(v_out_disp):.2f}\\text{{ V}}$):** "
            f"Initial gain $A_{{v0}} = {av_small:.2f} > 3$ starts oscillation, and Zener diodes $D_1, D_2$ regulate effective gain to $3.00$ (THD = {thd_pct:.2f}%)."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.12, 1.38], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-10*)**")
        fig_sch = draw_wien_bridge_schematic(r_kohm, c_uf, ri_kohm, rf1_kohm, rf2_kohm, use_zener, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Feedback Fraction:** $\\beta = 1/3$ at $f_o = \\frac{{1}}{{2\\pi R C}} = {fo:.1f}\\text{{ Hz}}$; "
            f"**Total $R_F = R_{{F1}} + R_{{F2}} = {rf1_kohm + rf2_kohm:.1f}\\text{{ k}}\\Omega$."
        )

    with cro_col:
        st.markdown("**Bridge Feedback ($v_+$) & Stabilized Output ($v_o$) (CRO)**")
        fig_cro, (ax_fb, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Non-inverting bridge feedback voltage v+(t)
        ax_fb.plot(t_ms, v_plus_disp, color="#1f77b4", linewidth=2.0, label="Bridge Feedback $v_+(t)$ (Pin 3)")
        y_lim_fb = max(np.max(np.abs(v_plus_disp)) * 1.35, 0.5)
        ax_fb.set_ylim(-y_lim_fb, y_lim_fb)
        style_cro_axes(
            ax_fb,
            title=f"CH1: Lead-Lag Bridge Feedback $v_+(t)$ ($0^\\circ$ phase shift at $f_o$)",
            xlabel="",
            ylabel="$v_+$ (V)"
        )
        ax_fb.legend(loc="upper right", fontsize=8.5)

        # Channel 2: Stabilized Sine Wave Output v_o(t)
        ax_out.plot(t_ms, v_out_disp, color="#d62728", linewidth=2.2, label="Oscillator Output $v_o(t)$ (Pin 6)")
        ax_out.axhline(vsat, color="#888888", linestyle=":", linewidth=1, label=f"$\\pm V_{{sat}} = \\pm {vsat:.1f}$ V")
        ax_out.axhline(-vsat, color="#888888", linestyle=":", linewidth=1)
        ax_out.set_ylim(-vsat * 1.2, vsat * 1.2)
        style_cro_axes(
            ax_out,
            title=f"CH2: Wien Bridge Sine Output $v_o(t)$ ($f_o = {fo:.1f}$ Hz, $V_{{o(pp)}} = {np.ptp(v_out_disp):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8.5)

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
            "Select three Wien resistor values ($R_a, R_b, R_c$) with fixed $C = "
            f"{c_uf:.3f}\\text{{ µF}}$, $R_i = {ri_kohm:.1f}\\text{{ k}}\\Omega$, $R_{{F1}} = {rf1_kohm:.1f}\\text{{ k}}\\Omega$, and $R_{{F2}} = {rf2_kohm:.1f}\\text{{ k}}\\Omega$ "
            "to compare the stabilized sine-wave outputs across **three different frequencies**:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            ra_kohm = st.number_input("Trial 1: Resistor $R_a$ (kΩ)", min_value=1.0, max_value=100.0, value=7.96, step=0.5)
        with fcol2:
            rb_kohm = st.number_input("Trial 2: Resistor $R_b$ (kΩ)", min_value=1.0, max_value=100.0, value=15.9, step=0.5)
        with fcol3:
            rc_kohm = st.number_input("Trial 3: Resistor $R_c$ (kΩ)", min_value=1.0, max_value=100.0, value=31.8, step=0.5)

        trials = [
            ("Trial 1 ($f_1$)", ra_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", rb_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", rc_kohm * 1e3, "#d62728"),
        ]

        # Common time axis covering 4 cycles of the lowest-frequency (largest R) trial
        max_R = max(ra_kohm, rb_kohm, rc_kohm) * 1e3
        T_max_trial = 2.0 * np.pi * max_R * C
        t_common_span = 4.0 * T_max_trial

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r_val, color) in enumerate(trials):
            fo_i = 1.0 / (2.0 * np.pi * r_val * C)
            T_i = 1.0 / fo_i
            cycles_needed = max(4, int(np.ceil(t_common_span / T_i)))
            t_i, _, vout_i, _, _, _, _, thd_i, vsat_i = simulate_wien_oscillator(
                r_val, C, Ri, Rf1, Rf2, use_zener, vz, vcc, num_cycles=cycles_needed, num_pts=4000
            )
            mask_i = t_i <= t_common_span
            ax = axes_cmp[idx]
            ax.plot(t_i[mask_i] * 1000.0, vout_i[mask_i], color=color, linewidth=2.0)
            ax.set_ylim(-vsat_i * 1.2, vsat_i * 1.2)
            style_cro_axes(
                ax,
                title=f"{label}: $R = {r_val/1e3:.2f}$ kΩ, $C = {c_uf:.3f}$ µF  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                xlabel="Time (ms)" if idx == 2 else "",
                ylabel="$v_o$ (V)"
            )
            table_rows.append({
                "Trial": label,
                "Wien Resistor R (kΩ)": f"{r_val/1e3:.2f}",
                "Wien Capacitor C (µF)": f"{c_uf:.3f}",
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
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7.4*)

            1. **Barkhausen Criterion & Frequency of Oscillation:**
               The Wien bridge consists of a series $RC$ branch and a parallel $RC$ branch connected to the non-inverting terminal (`+`), forming a lead-lag network whose feedback factor $\\beta = \\frac{{v_+}}{{v_o}}$ is:
               $$\\beta(f) = \\frac{{1}}{{3 + j\\left(\\frac{{f}}{{f_o}} - \\frac{{f_o}}{{f}}\\right)}}$$
               At the resonant frequency **$f_o$**, the reactive term vanishes, giving **zero phase shift ($0^\\circ$)** and **$\beta = \\frac{{1}}{{3}}$**:
               $$f_o = \\frac{{1}}{{2\\pi R C}} = {fo:.2f}\\text{{ Hz}}$$
               To satisfy the Barkhausen loop-gain condition $|A_v \\beta| \\ge 1$, the non-inverting amplifier gain must satisfy:
               $$A_v = 1 + \\frac{{R_F}}{{R_i}} > 3 \\implies R_F > 2 R_i$$

            2. **Amplitude Stabilization using Back-to-Back Zener Diodes ($D_1, D_2$):**
               * If $A_v$ is fixed permanently above $3$, the sinusoidal amplitude grows until the op-amp clips at $\\pm V_{{sat}}$, causing severe harmonic distortion.
               * In Gayakwad’s amplitude-stabilized circuit (Fig. 7-10), the feedback resistor is split into **$R_{{F1}}$** in series with **$R_{{F2}} \\parallel (D_1, D_2)$** such that:
                 $$1 + \\frac{{R_{{F1}} + R_{{F2}}}}{{R_i}} > 3 \\quad \\text{{and}} \\quad 1 + \\frac{{R_{{F1}}}}{{R_i}} < 3$$
               * When the output swing reaches the Zener breakdown threshold ($V_Z + 0.7\\text{{ V}}$ across $R_{{F2}}$), the diodes conduct and shunt $R_{{F2}}$, automatically reducing the effective gain to $A_v = 3.00$ and locking in a pure, low-distortion sine wave.
            """
        )