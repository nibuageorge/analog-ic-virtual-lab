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
    "Design and construct a sawtooth wave generator using **$\mu$A741** comparator and integrator"
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Textbook Sawtooth Generator Schematic
# -------------------------------------------------------------------------
def draw_sawtooth_gen_schematic(r1_kohm, c1_uf, r2_kohm, r3_kohm, r4_kohm, use_zener, vz, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.5)

        # --- Stage 1: Non-Inverting Comparator A1 ---
        a1 = d.add(elm.Opamp(leads=True).at((0, 0)).label('$A_1$', loc='center', fontsize=9))
        d.add(elm.Line().up().at(a1.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a1.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Inverting input (in1, top) grounded
        d.add(elm.Line().left().at(a1.in1).length(0.55))
        d.add(elm.Ground())

        # Non-inverting input (in2, bottom) -> Node P
        d.add(elm.Line().left().at(a1.in2).length(0.75))
        d.add(elm.Line().down().length(1.15))
        node_p = d.add(elm.Dot())
        px, py = node_p.center

        # Output of A1 -> First tap (BEFORE Zener diodes) for R3 feedback
        d.add(elm.Line().right().at(a1.out).length(0.45))
        r3_tap = d.add(elm.Dot())
        tap_x, tap_y = r3_tap.center

        # Rheostat / Potentiometer R3 connected between Node P and r3_tap (BEFORE diodes)
        d.add(elm.Potentiometer().right().at((px, py)).tox(tap_x).label(f'$R_3$\n{r3_kohm:.0f} kΩ', loc='bottom', fontsize=8))
        d.add(elm.Line().up().at((tap_x, py)).toy(tap_y))

        # Continue horizontally to the right -> Node vo' (Zener branch + v'_o tap)
        d.add(elm.Line().right().at((tap_x, tap_y)).length(0.95))
        vo1_node = d.add(elm.Dot())
        vo1_x, vo1_y = vo1_node.center

        # Tap going UP to v'_o square-wave indicator
        d.add(elm.Line().up().at((vo1_x, vo1_y)).length(0.85))
        d.add(elm.Line().right().length(0.35).label("$v'_o$", loc='top', fontsize=9))

        # Back-to-Back Zener Clamp D1 & D2 to Ground (completely clear to the right of R3)
        d.add(elm.Line().down().at((vo1_x, vo1_y)).length(0.3))
        if use_zener:
            d.add(elm.Zener().down().length(0.95).label('$D_1$', loc='right', fontsize=8.5))
            d.add(elm.Zener().down().reverse().length(0.95).label('$D_2$', loc='right', fontsize=8.5))
        else:
            d.add(elm.Line().down().length(1.9).linestyle('--'))
        d.add(elm.Ground())

        # --- Integrator Input Resistor R1 ---
        sum_x = vo1_x + 2.55
        d.add(elm.Resistor().right().at((vo1_x, vo1_y)).tox(sum_x).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        sum_node = d.add(elm.Dot())

        # --- Stage 2: Integrator A2 ---
        a2 = d.add(elm.Opamp(leads=True).anchor('in1').at((sum_x, vo1_y)).label('$A_2$', loc='center', fontsize=9))
        d.add(elm.Line().up().at(a2.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a2.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Feedback Capacitor C1 across A2
        d.add(elm.Line().up().at((sum_x, vo1_y)).length(1.15))
        d.add(elm.Capacitor().right().tox(a2.out[0]).label(f'$C_1$\n{c1_uf:.3f} µF', loc='top', fontsize=8.5))
        d.add(elm.Line().down().toy(a2.out[1]))
        vo_node = d.add(elm.Dot())
        vo_x, vo_y = vo_node.center

        # Final Sawtooth Output v_o
        d.add(elm.Line().right().at((vo_x, vo_y)).length(0.8).label('$v_o$', loc='right', fontsize=9.5))

        # --- Duty Cycle Adjust Potentiometer R4 on Non-Inverting Pin (+) of A2 ---
        in2_x, in2_y = a2.in2
        pot_y = in2_y - 1.15
        d.add(elm.Line().left().at((in2_x, in2_y)).length(0.45))
        wiper_x = d.here[0]
        d.add(elm.Line().down().toy(pot_y + 0.25))

        d.add(elm.Potentiometer().right().at((wiper_x - 0.85, pot_y)).length(1.7).label(f'$R_4$ ({r4_kohm:.0f} kΩ)\nDuty cycle adjust', loc='bottom', fontsize=8))
        d.add(elm.Line().left().at((wiper_x - 0.85, pot_y)).length(0.35).label(f'+{vcc:.0f}V', loc='left', fontsize=7.5))
        d.add(elm.Line().right().at((wiper_x + 0.85, pot_y)).length(0.35).label(f'-{vcc:.0f}V', loc='right', fontsize=7.5))

        # --- Feedback Resistor R2 from Node P down and across to final Output v_o ---
        fb_y = pot_y - 1.15
        d.add(elm.Resistor().down().at((px, py)).toy(fb_y).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='left', fontsize=8.5))
        d.add(elm.Line().right().tox(vo_x))
        d.add(elm.Line().up().toy(vo_y))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Generate Exact Analytical Waveforms for Textbook Sawtooth Circuit
# -------------------------------------------------------------------------
def simulate_sawtooth_generator(R1, C1, R2, R3, V_ref, use_zener, vz, vcc, t_span, num_pts=4000):
    vsat = max(vcc - 1.5, 0.5)
    v_clamp = min(vz + 0.7, vsat) if use_zener else vsat

    v_peak_ideal = (R2 / R3) * v_clamp
    vo_pp_ideal = 2.0 * v_peak_ideal

    if abs(V_ref) >= 0.96 * v_clamp:
        return None  # Latch-up if |V_ref| >= v_clamp

    slope_up = (v_clamp + V_ref) / (R1 * C1)
    slope_down = (v_clamp - V_ref) / (R1 * C1)

    T_rise = vo_pp_ideal / slope_up
    T_fall = vo_pp_ideal / slope_down
    T = T_rise + T_fall
    fo = 1.0 / T
    duty_pct = (T_rise / T) * 100.0

    t = np.linspace(0.0, t_span, num_pts)
    t_mod = np.mod(t, T)

    v_o_prime = np.empty_like(t)
    v_o_ideal = np.empty_like(t)

    # First interval [0, T_rise): v'_o = -v_clamp -> v_o rises linearly from -Vp to +Vp
    mask1 = t_mod < T_rise
    v_o_prime[mask1] = -v_clamp
    v_o_ideal[mask1] = -v_peak_ideal + slope_up * t_mod[mask1]

    # Second interval [T_rise, T): v'_o = +v_clamp -> v_o falls linearly from +Vp to -Vp
    mask2 = ~mask1
    v_o_prime[mask2] = v_clamp
    v_o_ideal[mask2] = v_peak_ideal - slope_down * (t_mod[mask2] - T_rise)

    v_o_disp = np.clip(v_o_ideal, -vsat, vsat)
    return t, v_o_prime, v_o_disp, fo, T, T_rise, T_fall, duty_pct, v_peak_ideal, vo_pp_ideal, v_clamp, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Integrator & Duty Cycle Adjust")
    r1_kohm = st.slider(
        "Integrator Resistor $R_1$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Default textbook value: 10 kΩ."
    )
    c1_uf = st.slider(
        "Integrator Capacitor $C_1$ (µF)",
        min_value=0.005,
        max_value=0.500,
        value=0.050,
        step=0.005,
        format="%.3f",
        help="Default textbook value: 0.05 µF."
    )
    r4_kohm = st.slider(
        "Duty-Cycle Potentiometer $R_4$ (kΩ)",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=5.0,
        help="Default textbook value: 20 kΩ connected between +VCC and -VEE."
    )
    wiper_pct = st.slider(
        "Potentiometer $R_4$ Wiper Position (%)",
        min_value=-85,
        max_value=85,
        value=-35,
        step=5,
        help="0% = symmetric triangular wave. Moving toward -85% or +85% creates a sawtooth wave."
    )

    st.subheader("2. Comparator Stage ($R_2, R_3$ & Zeners)")
    r2_kohm = st.slider(
        "Feedback Resistor $R_2$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="Default textbook value: 10 kΩ."
    )
    r3_kohm = st.slider(
        "Comparator Potentiometer $R_3$ (kΩ)",
        min_value=5.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        help="Default textbook value: 40 kΩ (50-kΩ pot). Must satisfy R3 > R2."
    )
    use_zener = st.checkbox(
        "Include Zener Clamp ($D_1, D_2$)",
        value=True,
        help="Clamps comparator output v'_o to ±(Vz + 0.7V) for stable amplitude."
    )
    vz = st.slider(
        "Zener Breakdown Voltage $V_Z$ (V)",
        min_value=3.3,
        max_value=12.0,
        value=5.1,
        step=0.1,
        disabled=not use_zener
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=2, max_value=8, value=4, step=1)

# -------------------------------------------------------------------------
# Primary Simulation Run
# -------------------------------------------------------------------------
R1 = r1_kohm * 1e3
C1 = c1_uf * 1e-6
R2 = r2_kohm * 1e3
R3 = r3_kohm * 1e3
V_ref = (wiper_pct / 100.0) * vcc

res = simulate_sawtooth_generator(R1, C1, R2, R3, V_ref, use_zener, vz, vcc, t_span=1e-3, num_pts=10)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    if res is None:
        st.error(
            f"**Integrator Latch-Up ($|V_{{ref}}| = {abs(V_ref):.2f}\\text{{ V}} \\ge V_{{clamp}} = {min(vz+0.7, max(vcc-1.5,0.5)):.2f}\\text{{ V}}$):** "
            f"The DC reference voltage on Pin 3 of $A_2$ from potentiometer $R_4$ exceeds the clamped comparator swing, "
            f"preventing the integrator from reversing direction. Move the $R_4$ wiper closer to center ($0\\%$)."
        )
        fig_sch = draw_sawtooth_gen_schematic(r1_kohm, c1_uf, r2_kohm, r3_kohm, r4_kohm, use_zener, vz, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
    else:
        T_est = res[4]
        t_disp, v_o_prime, v_o_disp, fo, T, T_rise, T_fall, duty_pct, v_peak_ideal, vo_pp_ideal, v_clamp, vsat = simulate_sawtooth_generator(
            R1, C1, R2, R3, V_ref, use_zener, vz, vcc, t_span=num_cycles * T_est
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
        m2.metric("Rise / Fall ($T_1 / T_2$)", f"{T_rise*1000:.2f} / {T_fall*1000:.2f} ms")
        m3.metric("Sawtooth $V_{o(pp)}$", f"{np.ptp(v_o_disp):.2f} V")
        m4.metric("Duty Cycle ($T_1 / T$)", f"{duty_pct:.1f}%")

        if R2 >= R3:
            st.error(
                f"**Design Constraint Violated ($R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega \\ge R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$):** "
                f"Requires **$R_3 > R_2$** so that $V_p = (R_2/R_3)V_{{clamp}} < V_{{sat}}$."
            )
        elif abs(wiper_pct) <= 5:
            st.info(
                f"**Symmetric Triangular Mode ($V_{{ref}} = {V_ref:+.2f}\\text{{ V}}$ near $0\\text{{ V}}$):** "
                f"With potentiometer $R_4$ centered, rise and fall times are equal ($50\\%$). Move the $R_4$ wiper toward $-85\\%$ or $+85\\%$ to generate a sawtooth waveform."
            )
        else:
            st.success(
                f"**Sawtooth Wave Generation ($f_o = {fo:.2f}\\text{{ Hz}}$, $V_{{ref}} = {V_ref:+.2f}\\text{{ V}}$):** "
                f"Rise time $T_1 = {T_rise*1000:.2f}\\text{{ ms}}$ and fall time $T_2 = {T_fall*1000:.2f}\\text{{ ms}}$ (Duty cycle = {duty_pct:.1f}%)."
            )

        # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
        schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

        with schem_col:
            st.markdown("**Circuit Diagram (*Gayakwad*)**")
            fig_sch = draw_sawtooth_gen_schematic(r1_kohm, c1_uf, r2_kohm, r3_kohm, r4_kohm, use_zener, vz, vcc)
            st.pyplot(fig_sch)
            plt.close(fig_sch)
            st.caption(
                f"**$R_4$ Wiper Voltage:** $V_{{ref}} = {V_ref:+.2f}\\text{{ V}}$ on Pin 3 (+) of $A_2$; "
                f"**Comparator Swing:** $v'_o = \\pm {v_clamp:.2f}\\text{{ V}}$."
            )

        with cro_col:
            st.markdown("**Comparator ($v'_o$) & Sawtooth ($v_o$) Waveforms (CRO)**")
            fig_cro, (ax_sq, ax_saw) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
            t_ms = t_disp * 1000.0

            # Channel 1: Comparator Rectangular Wave v'_o(t)
            ax_sq.plot(t_ms, v_o_prime, color="#1f77b4", linewidth=2.1, label="Comparator Output $v'_o(t)$")
            ax_sq.set_ylim(-vsat * 1.25, vsat * 1.25)
            style_cro_axes(
                ax_sq,
                title=f"CH1: Comparator Pulse $v'_o(t)$ ($\\pm {v_clamp:.2f}$ V, Duty = {duty_pct:.1f}%)",
                xlabel="",
                ylabel="$v'_o$ (V)"
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

        # -----------------------------------------------------------------
        # Secondary Tabs: Multi-Frequency Comparison (3 Frequencies) + Theory
        # -----------------------------------------------------------------
        tab_multi, tab_theory = st.tabs([
            "🔁 Repeat for Three Frequencies ($f_1, f_2, f_3$)",
            "📖 Theory & Design Procedure (Gayakwad)"
        ])

        with tab_multi:
            st.markdown(
                "Select three integrator resistor values ($R_{1a}, R_{1b}, R_{1c}$) with fixed $C_1 = "
                f"{c1_uf:.3f}\\text{{ µF}}$, $R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega$, $R_3 = {r3_kohm:.0f}\\text{{ k}}\\Omega$, and $V_{{ref}} = {V_ref:+.2f}\\text{{ V}}$ "
                "to compare the sawtooth outputs across **three different frequencies**:"
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

            max_R1 = max(r1a_kohm, r1b_kohm, r1c_kohm) * 1e3
            res_max = simulate_sawtooth_generator(max_R1, C1, R2, R3, V_ref, use_zener, vz, vcc, t_span=1e-3, num_pts=10)
            t_common_span = 4.0 * res_max[4]

            fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
            table_rows = []

            for idx, (label, r1_val, color) in enumerate(trials):
                t_i, _, vout_i, fo_i, T_i, t1_i, t2_i, _, _, _, _, vsat_i = simulate_sawtooth_generator(
                    r1_val, C1, R2, R3, V_ref, use_zener, vz, vcc, t_span=t_common_span, num_pts=4000
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
                    "Resistor R1 (kΩ)": f"{r1_val/1e3:.1f}",
                    "Rise T1 (ms)": f"{t1_i*1000:.3f}",
                    "Fall T2 (ms)": f"{t2_i*1000:.3f}",
                    "Period T (ms)": f"{T_i*1000:.3f}",
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
                ### Theory & Design Procedure (*R. A. Gayakwad, Section 7.8*)

                1. **Principle of Operation:**
                   * The sawtooth-wave generator is obtained from the triangular-wave generator by connecting a **$20\\text{{ k}}\\Omega$ potentiometer ($R_4$, *Duty cycle adjust*)** between $+V_{{CC}}$ and $-V_{{EE}}$, with its wiper connected to the non-inverting input (`+`) of integrator **$A_2$**.
                   * Back-to-back Zener diodes **$D_1$ and $D_2$** clamp the comparator ($A_1$) square-wave output $v'_o$ at $\\pm V_Z$ (where $V_Z = V_{{Z1}} + 0.7\\text{{ V}}$), ensuring a well-defined, constant peak-to-peak output amplitude.
                   * When the wiper of $R_4$ is at the center ($0\\text{{ V}}$), the rise and fall times are equal ($T_1 = T_2$), yielding a symmetric **triangular wave**. Moving the wiper toward $+V_{{CC}}$ or $-V_{{EE}}$ applies a DC offset voltage to $A_2$'s `+` input, making the charge and discharge rates unequal ($T_1 \\ne T_2$) and converting the output into a **sawtooth waveform**.

                2. **Peak-to-Peak Output Voltage ($V_{{o(pp)}}$):**
                   $$V_{{o(pp)}} = 2 \\left(\\frac{{R_2}}{{R_3}}\\right) V_Z = {vo_pp_ideal:.2f}\\text{{ V}} \\quad (\\text{{with }} R_3 > R_2)$$
                   *(If Zener diodes $D_1, D_2$ are omitted, $V_Z$ is replaced by $V_{{sat}}$.)*

                3. **Nominal Frequency of Oscillation ($f_o$):**
                   When $R_4$ is centered ($V_{{ref}} = 0\\text{{ V}}$), the frequency is given by Gayakwad’s standard equation:
                   $$f_o = \\frac{{R_3}}{{4 R_1 C_1 R_2}} = {(R3 / (4.0 * R1 * C1 * R2)):.2f}\\text{{ Hz}}$$
                   *(Offsetting the wiper of $R_4$ toward $+V_{{CC}}$ or $-V_{{EE}}$ adjusts the duty cycle to produce the sawtooth ramp, shifting the actual operating frequency to **$f_o = {fo:.2f}\\text{{ Hz}}$**.)*
                """
            )
