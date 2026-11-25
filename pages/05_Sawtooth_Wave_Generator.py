import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 5: Sawtooth Wave Generator", layout="wide")
render_sidebar_header()

st.title("Experiment 5: Sawtooth Wave Generator using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct a sawtooth wave generator using **$\mu$A741** comparator ($A_1$), integrator ($A_2$), "
    "and diode-resistor steering ($D_1, R_4$), study the output waveform, and repeat for three frequencies "
    "(*Gayakwad, Section 7.8*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Sawtooth Wave Generator Schematic (Fig. 7-17)
# -------------------------------------------------------------------------
def draw_sawtooth_gen_schematic(r1_kohm, r4_ohm, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc, polarity):
    r4_label = f"{r4_ohm/1000:.2f} kΩ" if r4_ohm >= 1000 else f"{r4_ohm:.0f} Ω"
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.45, fontsize=8.6)

        # --- Stage 1: Non-Inverting Comparator (A1) ---
        a1 = d.add(elm.Opamp(leads=True).label('$A_1$\n($\mu$A741)', loc='center', fontsize=8))
        d.add(elm.Line().up().at(a1.vd).length(0.4).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a1.vs).length(0.4).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Ground Pin 2 of A1
        d.add(elm.Line().left().at(a1.in1).length(0.45))
        d.add(elm.Ground())

        # Summing node P at Pin 3 of A1
        d.add(elm.Line().left().at(a1.in2).length(0.75))
        node_p = d.add(elm.Dot().label('P', loc='top', fontsize=8.5))

        # Resistor R2 from node P going left to feedback return line
        d.add(elm.Resistor().left().at(node_p.center).length(1.65).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='top'))
        fb_left = d.here

        # Output of A1 (Asymmetric pulse v_o1)
        d.add(elm.Line().right().at(a1.out).length(0.65))
        vo1_node = d.add(elm.Dot().label('$v_{o1}$', loc='top', fontsize=9))

        # Resistor R3 from node P down and over to vo1_node (below -VCC label)
        d.add(elm.Line().down().at(node_p.center).length(1.65))
        d.add(elm.Resistor().right().tox(vo1_node.center).label(f'$R_3$\n{r3_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Line().up().toy(vo1_node.center))

        # --- Parallel Diode-Resistor Steering Network (R1 || (D1 + R4)) ---
        d.add(elm.Line().right().at(vo1_node.center).length(0.45))
        branch_left = d.add(elm.Dot())

        # Main sweep resistor R1 along primary axis
        d.add(elm.Resistor().right().at(branch_left.center).length(2.3).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='bottom'))
        branch_right = d.add(elm.Dot())

        # Parallel fast-retrace branch above R1: Diode D1 + Resistor R4
        d.add(elm.Line().up().at(branch_left.center).length(1.05))
        if polarity == "Positive-Going Ramp (Slow Rise, Fast Fall)":
            # Forward-biased when v_o1 = +Vsat -> fast negative-going fall
            d.add(elm.Diode().right().length(1.15).label('$D_1$', loc='top'))
        else:
            # Forward-biased when v_o1 = -Vsat -> fast positive-going rise
            d.add(elm.Diode().left().at((branch_left.center[0] + 1.15, branch_left.center[1] + 1.05)).length(1.15).label('$D_1$', loc='top'))
            d.here = (branch_left.center[0] + 1.15, branch_left.center[1] + 1.05)

        d.add(elm.Resistor().right().tox(branch_right.center).label(f'$R_4$\n{r4_label}', loc='top'))
        d.add(elm.Line().down().toy(branch_right.center))

        # --- Stage 2: Inverting Integrator (A2) ---
        d.add(elm.Line().right().at(branch_right.center).length(0.45))
        sum_node2 = d.add(elm.Dot())

        a2 = d.add(elm.Opamp(leads=True).anchor('in1').label('$A_2$\n($\mu$A741)', loc='center', fontsize=8))
        d.add(elm.Line().up().at(a2.vd).length(0.4).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a2.vs).length(0.4).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Offset Minimizing Resistor R_OM = R1 on Pin 3 of A2
        d.add(elm.Line().left().at(a2.in2).length(0.45))
        d.add(elm.Resistor().down().length(1.1).label(f'$R_{{OM}}$\n{rom_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Ground())

        # Feedback Capacitor C1 across A2
        d.add(elm.Line().up().at(sum_node2.center).length(1.15))
        d.add(elm.Capacitor().right().tox(a2.out).label(f'$C_1$\n{c1_uf:.3f} µF', loc='top'))
        d.add(elm.Line().down().toy(a2.out))
        vo_node = d.add(elm.Dot())

        # Final Sawtooth Output v_o
        d.add(elm.Line().right().at(vo_node.center).length(0.8).label('$v_o$ (Sawtooth)', loc='right'))

        # Closed-loop feedback path from v_o back to R2 (routed cleanly below R3)
        d.add(elm.Line().down().at(vo_node.center).length(2.55))
        d.add(elm.Line().left().tox(fb_left[0]))
        d.add(elm.Line().up().toy(fb_left[1]))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Generate Exact Analytical Waveforms for Sawtooth Wave Generator
# -------------------------------------------------------------------------
def simulate_sawtooth_generator(R1, R4, C1, R2, R3, vcc, polarity, t_span, num_pts=4000):
    vsat = max(vcc - 1.5, 0.5)
    vd = 0.7  # Silicon diode (1N4148 / 1N4007) forward drop

    v_peak_ideal = (R2 / R3) * vsat
    vo_pp_ideal = 2.0 * v_peak_ideal

    # Slow sweep path (D1 reverse-biased): current magnitude = Vsat / R1
    slope_sweep = vsat / (R1 * C1)
    T_sweep = vo_pp_ideal / slope_sweep

    # Fast retrace/flyback path (D1 forward-biased): R1 in parallel with R4
    v_drive_flyback = max(vsat - vd, 0.2)
    i_flyback = (vsat / R1) + (v_drive_flyback / R4)
    slope_flyback = i_flyback / C1
    T_flyback = vo_pp_ideal / slope_flyback

    T = T_sweep + T_flyback
    fo = 1.0 / T
    linearity_pct = (T_sweep / T) * 100.0

    t = np.linspace(0.0, t_span, num_pts)
    t_mod = np.mod(t, T)

    v_o1 = np.empty_like(t)
    v_o_ideal = np.empty_like(t)

    if polarity == "Positive-Going Ramp (Slow Rise, Fast Fall)":
        # [0, T_sweep): v_o1 = -Vsat (D1 OFF) -> v_o ramps linearly UP from -Vp to +Vp
        mask1 = t_mod < T_sweep
        v_o1[mask1] = -vsat
        v_o_ideal[mask1] = -v_peak_ideal + slope_sweep * t_mod[mask1]

        # [T_sweep, T): v_o1 = +Vsat (D1 ON) -> v_o falls rapidly from +Vp to -Vp
        mask2 = ~mask1
        v_o1[mask2] = vsat
        v_o_ideal[mask2] = v_peak_ideal - slope_flyback * (t_mod[mask2] - T_sweep)
    else:
        # [0, T_sweep): v_o1 = +Vsat (D1 OFF) -> v_o ramps linearly DOWN from +Vp to -Vp
        mask1 = t_mod < T_sweep
        v_o1[mask1] = vsat
        v_o_ideal[mask1] = v_peak_ideal - slope_sweep * t_mod[mask1]

        # [T_sweep, T): v_o1 = -Vsat (D1 ON) -> v_o rises rapidly from -Vp to +Vp
        mask2 = ~mask1
        v_o1[mask2] = -vsat
        v_o_ideal[mask2] = -v_peak_ideal + slope_flyback * (t_mod[mask2] - T_sweep)

    v_o_disp = np.clip(v_o_ideal, -vsat, vsat)
    return t, v_o1, v_o_disp, fo, T, T_sweep, T_flyback, linearity_pct, v_peak_ideal, vo_pp_ideal, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Ramp & Flyback Components")
    polarity = st.radio(
        "Sawtooth Ramp Polarity (Diode $D_1$ Orientation)",
        [
            "Positive-Going Ramp (Slow Rise, Fast Fall)",
            "Negative-Going Ramp (Slow Fall, Fast Rise)"
        ]
    )
    r1_kohm = st.slider(
        "Sweep Resistor $R_1$ (kΩ)",
        min_value=2.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Sets the linear ramp duration T1 = 2·R1·C1·(R2/R3)."
    )
    r4_ohm = st.slider(
        "Flyback Resistor $R_4$ (Ω)",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
        help="In series with D1; smaller R4 yields faster flyback/retrace (R4 << R1)."
    )
    c1_uf = st.slider(
        "Integrator Capacitor $C_1$ (µF)",
        min_value=0.005,
        max_value=0.500,
        value=0.050,
        step=0.005,
        format="%.3f"
    )

    st.subheader("2. Comparator Stage ($A_1$: $R_2, R_3$)")
    r2_kohm = st.slider(
        "Feedback Resistor $R_2$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Sets peak-to-peak sawtooth amplitude Vo(pp) = 2·(R2/R3)·Vsat."
    )
    r3_kohm = st.slider(
        "Comparator Resistor $R_3$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=22.0,
        step=0.5,
        help="Must satisfy R3 > R2 so sawtooth peak stays below Vsat."
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=2, max_value=8, value=4, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R1 = r1_kohm * 1e3
R4 = float(r4_ohm)
C1 = c1_uf * 1e-6
R2 = r2_kohm * 1e3
R3 = r3_kohm * 1e3
rom_kohm = r1_kohm

# Preliminary period estimate for display window
_, _, _, _, T_est, _, _, _, _, _, _ = simulate_sawtooth_generator(
    R1, R4, C1, R2, R3, vcc, polarity, t_span=1e-3, num_pts=10
)
t_disp, v_o1, v_o_disp, fo, T, T_sweep, T_flyback, linearity_pct, v_peak_ideal, vo_pp_ideal, vsat = simulate_sawtooth_generator(
    R1, R4, C1, R2, R3, vcc, polarity, t_span=num_cycles * T_est
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Ramp / Flyback ($T_1 / T_2$)", f"{T_sweep*1000:.2f} / {T_flyback*1000:.2f} ms")
    m3.metric("Sawtooth $V_{o(pp)}$", f"{np.ptp(v_o_disp):.2f} V")
    m4.metric("Sweep Duty Ratio", f"{linearity_pct:.1f}%")

    if R2 >= R3:
        st.error(
            f"**Design Constraint Violated ($R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega \\ge R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$):** "
            f"Gayakwad requires **$R_3 > R_2$** so that $V_p = (R_2/R_3)V_{{sat}} < V_{{sat}}$."
        )
    elif R4 >= 0.25 * R1:
        st.warning(
            f"**Slow Flyback Notice ($R_4 = {format_eng(R4, 'Ω')}$ is relatively large compared to $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$):** "
            f"For a sharp sawtooth retrace, Gayakwad recommends **$R_4 \\ll R_1$**."
        )
    else:
        st.success(
            f"**Sawtooth Wave Generation ($f_o = {fo:.2f}\\text{{ Hz}}$, $V_{{o(pp)}} = {vo_pp_ideal:.2f}\\text{{ V}}$):** "
            f"Linear sweep time $T_1 = {T_sweep*1000:.2f}\\text{{ ms}}$ ({linearity_pct:.1f}% of period) and fast flyback $T_2 = {T_flyback*1000:.3f}\\text{{ ms}}$."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-17*)**")
        fig_sch = draw_sawtooth_gen_schematic(r1_kohm, r4_ohm, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc, polarity)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Steering Branch:** $R_1 = {r1_kohm:.1f}\\text{{ k}}\\Omega$ (ramp) in parallel with $D_1 + R_4$ ($R_4 = {format_eng(R4, 'Ω')}$, flyback)."
        )

    with cro_col:
        st.markdown("**Comparator & Sawtooth Waveforms (CRO)**")
        fig_cro, (ax_sq, ax_saw) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Asymmetric Rectangular Wave v_o1(t)
        ax_sq.plot(t_ms, v_o1, color="#1f77b4", linewidth=2.1, label="Comparator Output $v_{o1}(t)$")
        ax_sq.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_sq,
            title=f"CH1: Comparator Pulse $v_{{o1}}(t)$ (Asymmetric Duty Cycle)",
            xlabel="",
            ylabel="$v_{o1}$ (V)"
        )
        ax_sq.legend(loc="upper right", fontsize=8.5)

        # Channel 2: Sawtooth Wave Output v_o(t)
        ax_saw.plot(t_ms, v_o_disp, color="#d62728", linewidth=2.2, label="Sawtooth Output $v_o(t)$")
        if R2 < R3:
            ax_saw.axhline(v_peak_ideal, color="#2ca02c", linestyle="--", linewidth=1, label=f"$+V_p = +{v_peak_ideal:.2f}$ V")
            ax_saw.axhline(-v_peak_ideal, color="#ff7f0e", linestyle="--", linewidth=1, label=f"$-V_p = -{v_peak_ideal:.2f}$ V")
        ax_saw.set_ylim(-vsat * 1.2, vsat * 1.2)
        style_cro_axes(
            ax_saw,
            title=f"CH2: Sawtooth Output $v_o(t)$ ($f_o = {fo:.1f}$ Hz, $V_{{o(pp)}} = {np.ptp(v_o_disp):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_saw.legend(loc="upper right", fontsize=8, ncol=2)

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
            "Select three sweep resistor values ($R_{1a}, R_{1b}, R_{1c}$) with fixed $C_1 = "
            f"{c1_uf:.3f}\\text{{ µF}}$, $R_4 = {format_eng(R4, 'Ω')}$, $R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega$, and $R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$ "
            "to compare the sawtooth outputs across **three different frequencies** simultaneously:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            r1a_kohm = st.number_input("Trial 1: Sweep Resistor $R_{1a}$ (kΩ)", min_value=2.0, max_value=100.0, value=10.0, step=1.0)
        with fcol2:
            r1b_kohm = st.number_input("Trial 2: Sweep Resistor $R_{1b}$ (kΩ)", min_value=2.0, max_value=100.0, value=20.0, step=1.0)
        with fcol3:
            r1c_kohm = st.number_input("Trial 3: Sweep Resistor $R_{1c}$ (kΩ)", min_value=2.0, max_value=100.0, value=40.0, step=1.0)

        trials = [
            ("Trial 1 ($f_1$)", r1a_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", r1b_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", r1c_kohm * 1e3, "#d62728"),
        ]

        # Common time axis covering 4 cycles of the lowest-frequency (largest R1) trial
        max_R1 = max(r1a_kohm, r1b_kohm, r1c_kohm) * 1e3
        _, _, _, _, T_max_trial, _, _, _, _, _, _ = simulate_sawtooth_generator(
            max_R1, R4, C1, R2, R3, vcc, polarity, t_span=1e-3, num_pts=10
        )
        t_common_span = 4.0 * T_max_trial

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r1_val, color) in enumerate(trials):
            t_i, _, vout_i, fo_i, T_i, t1_i, t2_i, _, _, _, vsat_i = simulate_sawtooth_generator(
                r1_val, R4, C1, R2, R3, vcc, polarity, t_span=t_common_span, num_pts=4000
            )
            ax = axes_cmp[idx]
            ax.plot(t_i * 1000.0, vout_i, color=color, linewidth=2.0)
            ax.set_ylim(-vsat_i * 1.2, vsat_i * 1.2)
            style_cro_axes(
                ax,
                title=f"{label}: $R_1 = {r1_val/1e3:.1f}$ kΩ, $C_1 = {c1_uf:.3f}$ µF  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                xlabel="Time (ms)" if idx == 2 else "",
                ylabel="$v_o$ (V)"
            )
            table_rows.append({
                "Trial": label,
                "Sweep R1 (kΩ)": f"{r1_val/1e3:.1f}",
                "Ramp T1 (ms)": f"{t1_i*1000:.3f}",
                "Flyback T2 (ms)": f"{t2_i*1000:.3f}",
                "Total Period T (ms)": f"{T_i*1000:.3f}",
                "Frequency fo (Hz)": f"{fo_i:.2f}",
                "Sawtooth Vo(pp) (V)": f"{np.ptp(vout_i):.2f}"
            })

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("**Observation Table (Three Frequencies)**")
        st.table(table_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (*R. A. Gayakwad, Section 7.8*)

            1. **Conversion of Triangular to Sawtooth Wave:**
               A triangular wave generator has equal rise and fall times because the integrator charging and discharging currents are equal ($\\pm V_{{sat}} / R_1$). By connecting a **diode $D_1$ in series with a small resistor $R_4$** in parallel with $R_1$, the charge and discharge time constants become unequal:
               * **Linear Sweep Interval ($T_1$):** Diode $D_1$ is reverse-biased (OFF). Capacitor $C_1$ integrates slowly through $R_1$ alone:
                 $$T_1 = \\frac{{2 R_1 C_1 R_2}}{{R_3}} = {T_sweep*1000:.3f}\\text{{ ms}}$$
               * **Fast Flyback / Retrace Interval ($T_2$):** When comparator $A_1$ switches polarity, $D_1$ becomes forward-biased (ON), placing $R_4$ ($R_4 \\ll R_1$) in parallel with $R_1$ and discharging $C_1$ rapidly:
                 $$T_2 \\approx \\frac{{2 (R_1 \\parallel R_4) C_1 R_2}}{{R_3}} = {T_flyback*1000:.3f}\\text{{ ms}}$$

            2. **Peak-to-Peak Amplitude & Oscillation Frequency:**
               $$V_{{o(pp)}} = 2 V_p = 2 \\left(\\frac{{R_2}}{{R_3}}\\right) V_{{sat}} = {vo_pp_ideal:.2f}\\text{{ V}} \\quad (R_3 > R_2)$$
               $$T = T_1 + T_2 = {T*1000:.3f}\\text{{ ms}}, \\qquad f_o = \\frac{{1}}{{T}} = {fo:.2f}\\text{{ Hz}}$$
               *(For $R_4 \\ll R_1$, $T_2 \\ll T_1$ and $f_o \\approx \\frac{{R_3}}{{2 R_1 C_1 R_2}}$.)*

            3. **Reversing Ramp Polarity:**
               Reversing the orientation of diode **$D_1$** interchanges the slow-ramp and fast-retrace intervals, converting a positive-going sawtooth ramp into a negative-going sawtooth ramp.
            """
        )