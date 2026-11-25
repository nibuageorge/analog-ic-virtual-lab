import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 4: Triangular Wave Generator", layout="wide")
render_sidebar_header()

st.title("Experiment 4: Triangular Wave Generator using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct a two-stage triangular wave generator using **$\mu$A741** comparator and "
    "integrator, study the square wave ($v_{o1}$) and triangular wave ($v_o$) outputs "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Gayakwad Two-Stage Triangular Wave Generator (Fig. 7-15)
# -------------------------------------------------------------------------
def draw_triangular_gen_schematic(r1_kohm, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.8)

        # --- Stage 1: Non-Inverting Comparator (A1) ---
        a1 = d.add(elm.Opamp(leads=True).label('$A_1$\n($\mu$A741)', loc='center', fontsize=8))
        d.add(elm.Line().up().at(a1.vd).length(0.4).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a1.vs).length(0.4).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Ground Pin 2 of A1
        d.add(elm.Line().left().at(a1.in1).length(0.5))
        d.add(elm.Ground())

        # Summing node P at Pin 3 of A1
        d.add(elm.Line().left().at(a1.in2).length(0.8))
        node_p = d.add(elm.Dot().label('P', loc='top', fontsize=8.5))

        # Resistor R2 from node P going left to feedback return line
        d.add(elm.Resistor().left().at(node_p.center).length(1.8).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='top'))
        fb_left = d.here

        # Output of A1 (Square wave v_o1)
        d.add(elm.Line().right().at(a1.out).length(0.7))
        vo1_node = d.add(elm.Dot().label('$v_{o1}$', loc='top', fontsize=9))

        # Resistor R3 from node P down and over to vo1_node (well below -VCC label)
        d.add(elm.Line().down().at(node_p.center).length(1.65))
        d.add(elm.Resistor().right().tox(vo1_node.center).label(f'$R_3$\n{r3_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Line().up().toy(vo1_node.center))

        # --- Stage 2: Inverting Integrator (A2) ---
        d.add(elm.Resistor().right().at(vo1_node.center).length(1.8).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top'))
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

        # Final Triangular Output v_o
        d.add(elm.Line().right().at(vo_node.center).length(0.8).label('$v_o$ (Triangular)', loc='right'))

        # Closed-loop feedback path from v_o back to R2 (routed cleanly below R3)
        d.add(elm.Line().down().at(vo_node.center).length(2.55))
        d.add(elm.Line().left().tox(fb_left[0]))
        d.add(elm.Line().up().toy(fb_left[1]))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Generate Exact Analytical Waveforms for Triangular Wave Generator
# -------------------------------------------------------------------------
def simulate_triangular_generator(R1, C1, R2, R3, vcc, t_span, num_pts=3500):
    vsat = max(vcc - 1.5, 0.5)
    v_peak_ideal = (R2 / R3) * vsat
    vo_pp_ideal = 2.0 * v_peak_ideal

    T = (4.0 * R1 * C1 * R2) / R3
    fo = 1.0 / T

    t = np.linspace(0.0, t_span, num_pts)
    t_mod = np.mod(t, T)
    half_T = T / 2.0

    v_o1 = np.empty_like(t)
    v_o_ideal = np.empty_like(t)

    # First half-cycle [0, T/2): v_o1 = -Vsat -> Integrator ramps UP from -Vp to +Vp
    mask1 = t_mod < half_T
    t1 = t_mod[mask1]
    v_o1[mask1] = -vsat
    v_o_ideal[mask1] = -v_peak_ideal + (vsat / (R1 * C1)) * t1

    # Second half-cycle [T/2, T): v_o1 = +Vsat -> Integrator ramps DOWN from +Vp to -Vp
    mask2 = ~mask1
    t2 = t_mod[mask2] - half_T
    v_o1[mask2] = vsat
    v_o_ideal[mask2] = v_peak_ideal - (vsat / (R1 * C1)) * t2

    v_o_disp = np.clip(v_o_ideal, -vsat, vsat)
    return t, v_o1, v_o_disp, fo, T, v_peak_ideal, vo_pp_ideal, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Integrator Stage ($A_2$: $R_1, C_1$)")
    r1_kohm = st.slider(
        "Integrator Resistor $R_1$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Controls integration time constant R1·C1 and oscillation frequency fo."
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
        help="Sets peak-to-peak triangular output Vo(pp) = 2·(R2/R3)·Vsat."
    )
    r3_kohm = st.slider(
        "Comparator Resistor $R_3$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=22.0,
        step=0.5,
        help="Must satisfy R3 > R2 so triangular peak (R2/R3)·Vsat stays below Vsat."
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=2, max_value=8, value=4, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R1 = r1_kohm * 1e3
C1 = c1_uf * 1e-6
R2 = r2_kohm * 1e3
R3 = r3_kohm * 1e3
rom_kohm = r1_kohm

T_tmp = (4.0 * R1 * C1 * R2) / R3
t_disp, v_o1, v_o_disp, fo, T, v_peak_ideal, vo_pp_ideal, vsat = simulate_triangular_generator(
    R1, C1, R2, R3, vcc, t_span=num_cycles * T_tmp
)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
    m2.metric("Time Period ($T$)", format_eng(T, "s"))
    m3.metric("Triangular $V_{o(pp)}$", f"{np.ptp(v_o_disp):.2f} V")
    m4.metric("Square $V_{o1(pp)}$", f"{2*vsat:.1f} V")

    if R2 >= R3:
        st.error(
            f"**Design Constraint Violated ($R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega \\ge R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$):** "
            f"Gayakwad requires **$R_3 > R_2$** so that $V_{{p}} = (R_2/R_3)V_{{sat}} < V_{{sat}}$. "
            f"When $R_2 \\ge R_3$, Integrator $A_2$ saturates before Comparator $A_1$ can switch states!"
        )
    else:
        st.success(
            f"**Linear Triangular Wave Generation ($f_o = {fo:.2f}\\text{{ Hz}}$, $V_{{o(pp)}} = {vo_pp_ideal:.2f}\\text{{ V}}$):** "
            f"Comparator $A_1$ switches whenever $v_o(t)$ reaches $\\pm (R_2/R_3)V_{{sat}} = \\pm {v_peak_ideal:.2f}\\text{{ V}}$, "
            f"reversing the integration slope $\\pm V_{{sat}}/(R_1 C_1)$."
        )

    # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
    schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-15*)**")
        fig_sch = draw_triangular_gen_schematic(r1_kohm, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Stage $A_1$:** Non-inverting comparator ($R_2, R_3$). "
            f"**Stage $A_2$:** Inverting integrator ($R_1, C_1, R_{{OM}} = {rom_kohm:.1f}\\text{{ k}}\\Omega$)."
        )

    with cro_col:
        st.markdown("**Comparator & Integrator Waveforms (CRO)**")
        fig_cro, (ax_sq, ax_tri) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
        t_ms = t_disp * 1000.0

        # Channel 1: Comparator Square Wave v_o1(t)
        ax_sq.plot(t_ms, v_o1, color="#1f77b4", linewidth=2.1, label="Comparator Output $v_{o1}(t)$")
        ax_sq.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_sq,
            title=f"CH1: Comparator Square Wave $v_{{o1}}(t)$ ($\\pm V_{{sat}} = \\pm {vsat:.1f}$ V)",
            xlabel="",
            ylabel="$v_{o1}$ (V)"
        )
        ax_sq.legend(loc="upper right", fontsize=8.5)

        # Channel 2: Triangular Wave Output v_o(t)
        ax_tri.plot(t_ms, v_o_disp, color="#d62728", linewidth=2.2, label="Triangular Output $v_o(t)$")
        if R2 < R3:
            ax_tri.axhline(v_peak_ideal, color="#2ca02c", linestyle="--", linewidth=1, label=f"$+V_p = +{v_peak_ideal:.2f}$ V")
            ax_tri.axhline(-v_peak_ideal, color="#ff7f0e", linestyle="--", linewidth=1, label=f"$-V_p = -{v_peak_ideal:.2f}$ V")
        ax_tri.set_ylim(-vsat * 1.2, vsat * 1.2)
        style_cro_axes(
            ax_tri,
            title=f"CH2: Triangular Wave Output $v_o(t)$ ($f_o = {fo:.1f}$ Hz, $V_{{o(pp)}} = {np.ptp(v_o_disp):.2f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_tri.legend(loc="upper right", fontsize=8, ncol=2)

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
            "Select three integrator resistor values ($R_{1a}, R_{1b}, R_{1c}$) with fixed $C_1 = "
            f"{c1_uf:.3f}\\text{{ µF}}$, $R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega$, and $R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$ "
            "to compare the triangular wave outputs across **three different frequencies** simultaneously:"
        )

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            r1a_kohm = st.number_input("Trial 1: Resistor $R_{1a}$ (kΩ)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
        with fcol2:
            r1b_kohm = st.number_input("Trial 2: Resistor $R_{1b}$ (kΩ)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        with fcol3:
            r1c_kohm = st.number_input("Trial 3: Resistor $R_{1c}$ (kΩ)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

        trials = [
            ("Trial 1 ($f_1$)", r1a_kohm * 1e3, "#1f77b4"),
            ("Trial 2 ($f_2$)", r1b_kohm * 1e3, "#2ca02c"),
            ("Trial 3 ($f_3$)", r1c_kohm * 1e3, "#d62728"),
        ]

        # Common time axis covering 4 cycles of the lowest-frequency (largest R1) trial
        max_R1 = max(r1a_kohm, r1b_kohm, r1c_kohm) * 1e3
        t_common_span = 4.0 * ((4.0 * max_R1 * C1 * R2) / R3)

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
        table_rows = []

        for idx, (label, r1_val, color) in enumerate(trials):
            t_i, _, vout_i, fo_i, T_i, _, _, vsat_i = simulate_triangular_generator(
                r1_val, C1, R2, R3, vcc, t_span=t_common_span, num_pts=3500
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
                "Integrator R1 (kΩ)": f"{r1_val/1e3:.1f}",
                "Capacitor C1 (µF)": f"{c1_uf:.3f}",
                "Time Period T (ms)": f"{T_i*1000:.3f}",
                "Frequency fo (Hz)": f"{fo_i:.2f}",
                "Triangular Vo(pp) (V)": f"{np.ptp(vout_i):.2f}"
            })

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("**Observation Table (Three Frequencies)**")
        st.table(table_rows)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Procedure (*R. A. Gayakwad, Section 7.7*)

            1. **Circuit Operation:**
               The generator consists of two op-amp stages in a closed feedback loop:
               * **$A_1$** acts as a non-inverting comparator (Schmitt trigger) whose non-inverting input voltage at node **P** is given by superposition of $v_{{o1}}$ and $v_o$:
                 $$v_P = \\frac{{R_3}}{{R_2 + R_3}} v_o + \\frac{{R_2}}{{R_2 + R_3}} v_{{o1}}$$
               * When $v_{{o1}} = +V_{{sat}}$, integrator **$A_2$** produces a negative-going linear ramp $v_o(t)$ with slope $-V_{{sat}} / (R_1 C_1)$. When $v_o(t)$ reaches $-V_p = -\\frac{{R_2}}{{R_3}} V_{{sat}}$, $v_P$ crosses $0\\text{{ V}}$ and $A_1$ switches to $v_{{o1}} = -V_{{sat}}$.
               * Integrator $A_2$ then reverses direction and ramps linearly upward until $v_o(t)$ reaches $+V_p = +\\frac{{R_2}}{{R_3}} V_{{sat}}$, repeating the cycle.

            2. **Peak-to-Peak Output Voltage ($V_{{o(pp)}}$):**
               $$V_{{o(pp)}} = 2 V_p = 2 \\left(\\frac{{R_2}}{{R_3}}\\right) V_{{sat}} = {vo_pp_ideal:.2f}\\text{{ V}}$$
               *(Note: $R_3$ must be greater than $R_2$ so that $V_p < V_{{sat}}$.)*

            3. **Time Period & Frequency of Oscillation ($f_o$):**
               During each half-cycle ($T/2$), $v_o$ changes by $V_{{o(pp)}}$ at a constant rate of $V_{{sat}} / (R_1 C_1)$:
               $$\\frac{{T}}{{2}} = \\frac{{V_{{o(pp)}}}}{{V_{{sat}} / (R_1 C_1)}} = \\frac{{2 R_1 C_1 R_2}}{{R_3}} \\implies T = \\frac{{4 R_1 C_1 R_2}}{{R_3}} = {T*1000:.3f}\\text{{ ms}}$$
               $$f_o = \\frac{{1}}{{T}} = \\frac{{R_3}}{{4 R_1 C_1 R_2}} = {fo:.2f}\\text{{ Hz}}$$

            4. **Offset Minimizing Resistor ($R_{{OM}}$):**
               Connected between Pin 3 of integrator $A_2$ and ground:
               $$R_{{OM}} = R_1 = {rom_kohm:.1f}\\text{{ k}}\\Omega$$
            """
        )
