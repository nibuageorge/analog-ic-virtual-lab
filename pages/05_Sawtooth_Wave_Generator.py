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
    "and a DC offset potentiometer ($R_p, R_4$)—without diodes—study the output waveform, and repeat for three frequencies "
    "(*Gayakwad, Section 7.8, Fig. 7-17*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Fig. 7-17 Sawtooth Generator Schematic
# -------------------------------------------------------------------------
def draw_sawtooth_gen_schematic(r1_kohm, r4_kohm, rp_kohm, wiper_pct, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.5)

        # --- Stage 1: Comparator A1 ---
        a1 = d.add(elm.Opamp(leads=True).at((0, 0)).label('$A_1$\n($\mu$A741)', loc='center', fontsize=8))
        d.add(elm.Line().up().at(a1.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a1.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Inverting input (in1, top) grounded
        d.add(elm.Line().left().at(a1.in1).length(0.5))
        d.add(elm.Ground())

        # Non-inverting input (in2, bottom) -> Junction P
        d.add(elm.Line().left().at(a1.in2).length(0.9))
        node_p = d.add(elm.Dot().label('P', loc='top', fontsize=8.5))

        # Comparator output v_o1
        d.add(elm.Line().right().at(a1.out).length(0.6))
        vo1_node = d.add(elm.Dot().label('$v_{o1}$', loc='top', fontsize=9))

        # Resistor R3 connecting Junction P to v_o1 below A1
        d.add(elm.Line().down().at(node_p.center).length(1.6))
        r3_corner = d.here
        d.add(elm.Resistor().right().tox(vo1_node.center).label(f'$R_3$\n{r3_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Line().up().toy(vo1_node.center))

        # --- Integrator Input: R1 from v_o1 to Integrator Summing Node ---
        d.add(elm.Resistor().right().at(vo1_node.center).length(2.8).label(f'$R_1$\n{r1_kohm:.1f} kΩ', loc='bottom'))
        sum_node = d.add(elm.Dot())

        # --- Stage 2: Integrator A2 ---
        a2 = d.add(elm.Opamp(leads=True).anchor('in1').at(sum_node.center).label('$A_2$\n($\mu$A741)', loc='center', fontsize=8))
        d.add(elm.Line().up().at(a2.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(a2.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # Offset Minimizing Resistor R_OM on Pin 3 (in2) of A2
        d.add(elm.Line().left().at(a2.in2).length(0.5))
        d.add(elm.Resistor().down().length(1.1).label(f'$R_{{OM}}$\n{rom_kohm:.2f} kΩ', loc='bottom'))
        d.add(elm.Ground())

        # Feedback Capacitor C1 across A2 (above A2)
        d.add(elm.Line().up().at(sum_node.center).length(1.15))
        d.add(elm.Capacitor().right().tox(a2.out).label(f'$C_1$\n{c1_uf:.3f} µF', loc='top'))
        d.add(elm.Line().down().toy(a2.out))
        vo_node = d.add(elm.Dot())

        # Final Sawtooth Output v_o
        d.add(elm.Line().right().at(vo_node.center).length(0.8).label('$v_o$ (Sawtooth)', loc='right'))

        # --- DC Offset Potentiometer Rp + R4 (cleanly above R1, left of C1) ---
        # Draw R4 straight up from sum_node, then left to Potentiometer Rp
        d.add(elm.Line().up().at(sum_node.center).length(2.15))
        d.add(elm.Resistor().left().length(1.7).label(f'$R_4$\n{r4_kohm:.1f} kΩ', loc='top'))
        wiper_pt = d.here

        # Vertical Potentiometer / Voltage Divider Rp at wiper_pt[0] - 0.35
        pot_x = wiper_pt[0] - 0.35
        pot_y = wiper_pt[1]
        d.add(elm.Potentiometer().up().at((pot_x, pot_y - 0.65)).to((pot_x, pot_y + 0.65)).label(f'$R_p$\n{rp_kohm:.0f} kΩ', loc='left'))
        d.add(elm.Line().up().at((pot_x, pot_y + 0.65)).length(0.3).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at((pot_x, pot_y - 0.65)).length(0.3).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Closed-Loop Feedback Path via R2 along bottom rail ---
        d.add(elm.Line().down().at(r3_corner).length(1.0))
        d.add(elm.Resistor().right().tox(vo_node.center).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='bottom'))
        d.add(elm.Line().up().toy(vo_node.center))

        fig = d.draw(show=False).fig
        return fig
# -------------------------------------------------------------------------
# Helper: Generate Exact Analytical Waveforms for Gayakwad Fig. 7-17
# -------------------------------------------------------------------------
def simulate_sawtooth_generator(R1, R4, V_dc, C1, R2, R3, vcc, t_span, num_pts=4000):
    vsat = max(vcc - 1.5, 0.5)
    v_peak_ideal = (R2 / R3) * vsat
    vo_pp_ideal = 2.0 * v_peak_ideal

    # Current from square wave: ±Vsat / R1; Current from DC pot: V_dc / R4
    i_sq = vsat / R1
    i_dc = abs(V_dc) / R4

    # Condition for oscillation: |i_dc| < i_sq so integrator can reverse direction
    if i_dc >= 0.98 * i_sq:
        return None  # Latch-up condition

    # Two asymmetric integration slopes:
    # Fast slope (currents add): (i_sq + i_dc) / C1
    # Slow slope (currents oppose): (i_sq - i_dc) / C1
    slope_slow = (i_sq - i_dc) / C1
    slope_fast = (i_sq + i_dc) / C1

    T_ramp = vo_pp_ideal / slope_slow
    T_flyback = vo_pp_ideal / slope_fast
    T = T_ramp + T_flyback
    fo = 1.0 / T
    duty_pct = (T_ramp / T) * 100.0

    t = np.linspace(0.0, t_span, num_pts)
    t_mod = np.mod(t, T)

    v_o1 = np.empty_like(t)
    v_o_ideal = np.empty_like(t)

    if V_dc >= 0:
        # When V_dc > 0:
        # For v_o1 = -Vsat: net current into Pin 2 is (-Vsat/R1 + V_dc/R4) -> small negative current -> slow positive-going ramp!
        # For v_o1 = +Vsat: net current into Pin 2 is (+Vsat/R1 + V_dc/R4) -> large positive current -> fast negative-going fall!
        mask1 = t_mod < T_ramp
        v_o1[mask1] = -vsat
        v_o_ideal[mask1] = -v_peak_ideal + slope_slow * t_mod[mask1]

        mask2 = ~mask1
        v_o1[mask2] = vsat
        v_o_ideal[mask2] = v_peak_ideal - slope_fast * (t_mod[mask2] - T_ramp)
    else:
        # When V_dc < 0: slow negative-going ramp, fast positive-going flyback
        mask1 = t_mod < T_ramp
        v_o1[mask1] = vsat
        v_o_ideal[mask1] = v_peak_ideal - slope_slow * t_mod[mask1]

        mask2 = ~mask1
        v_o1[mask2] = -vsat
        v_o_ideal[mask2] = -v_peak_ideal + slope_fast * (t_mod[mask2] - T_ramp)

    v_o_disp = np.clip(v_o_ideal, -vsat, vsat)
    return t, v_o1, v_o_disp, fo, T, T_ramp, T_flyback, duty_pct, v_peak_ideal, vo_pp_ideal, vsat

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Integrator & DC Offset Potentiometer")
    r1_kohm = st.slider(
        "Integrator Resistor $R_1$ (kΩ)",
        min_value=2.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Sets base integration current Vsat / R1."
    )
    r4_kohm = st.slider(
        "Offset Series Resistor $R_4$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Connects potentiometer Rp wiper to integrator summing node (Pin 2 of A2)."
    )
    rp_kohm = st.slider(
        "Potentiometer $R_p$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )
    wiper_pct = st.slider(
        "Potentiometer Wiper Position (%)",
        min_value=-90,
        max_value=90,
        value=65,
        step=5,
        help="0% = center (symmetric triangular wave). Moving toward +90% or -90% creates a sawtooth wave."
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
R4 = r4_kohm * 1e3
C1 = c1_uf * 1e-6
R2 = r2_kohm * 1e3
R3 = r3_kohm * 1e3
rom_kohm = (r1_kohm * r4_kohm) / (r1_kohm + r4_kohm)  # R_OM = R1 || R4
V_dc = (wiper_pct / 100.0) * vcc

st.sidebar.caption(f"Potentiometer DC Wiper Voltage: **$V_{{dc}} = {V_dc:+.2f}\\text{{ V}}$**")

res = simulate_sawtooth_generator(R1, R4, V_dc, C1, R2, R3, vcc, t_span=1e-3, num_pts=10)

# -------------------------------------------------------------------------
# Right Display Panel: Circuit Diagram + Live CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    if res is None:
        st.error(
            f"**Integrator Latch-Up ($|V_{{dc}}| / R_4 \\ge V_{{sat}} / R_1$):** "
            f"The DC offset current from potentiometer $R_p$ ($|V_{{dc}}|/R_4 = {abs(V_dc)/R4*1e3:.2f}\\text{{ mA}}$) exceeds "
            f"the square-wave integration current ($V_{{sat}}/R_1 = {max(vcc-1.5,0.5)/R1*1e3:.2f}\\text{{ mA}}$), preventing the integrator from reversing slope. "
            f"Move the wiper closer to center or increase $R_4$."
        )
        fig_sch = draw_sawtooth_gen_schematic(r1_kohm, r4_kohm, rp_kohm, wiper_pct, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
    else:
        T_est = res[4]
        t_disp, v_o1, v_o_disp, fo, T, T_ramp, T_flyback, duty_pct, v_peak_ideal, vo_pp_ideal, vsat = simulate_sawtooth_generator(
            R1, R4, V_dc, C1, R2, R3, vcc, t_span=num_cycles * T_est
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Oscillation Frequency ($f_o$)", format_eng(fo, "Hz"))
        m2.metric("Ramp / Flyback ($T_1 / T_2$)", f"{T_ramp*1000:.2f} / {T_flyback*1000:.2f} ms")
        m3.metric("Sawtooth $V_{o(pp)}$", f"{np.ptp(v_o_disp):.2f} V")
        m4.metric("Sweep Duty Ratio", f"{duty_pct:.1f}%")

        if R2 >= R3:
            st.error(
                f"**Design Constraint Violated ($R_2 = {r2_kohm:.1f}\\text{{ k}}\\Omega \\ge R_3 = {r3_kohm:.1f}\\text{{ k}}\\Omega$):** "
                f"Gayakwad requires **$R_3 > R_2$** so that $V_p = (R_2/R_3)V_{{sat}} < V_{{sat}}$."
            )
        elif abs(wiper_pct) <= 5:
            st.info(
                f"**Symmetric Triangular Mode ($V_{{dc}} = {V_dc:+.2f}\\text{{ V}}$ near $0\\text{{ V}}$):** "
                f"With potentiometer $R_p$ centered, rise and fall times are equal ($50\\%$). Move the wiper toward $+90\\%$ or $-90\\%$ to generate a sawtooth waveform."
            )
        else:
            st.success(
                f"**Sawtooth Wave Generation ($f_o = {fo:.2f}\\text{{ Hz}}$, $V_{{dc}} = {V_dc:+.2f}\\text{{ V}}$):** "
                f"Linear ramp time $T_1 = {T_ramp*1000:.2f}\\text{{ ms}}$ ({duty_pct:.1f}% of period) and fast flyback $T_2 = {T_flyback*1000:.2f}\\text{{ ms}}$."
            )

        # Top row in main display: Circuit Diagram (Left) + CRO Waveforms (Right)
        schem_col, cro_col = st.columns([1.15, 1.35], gap="medium")

        with schem_col:
            st.markdown("**Circuit Diagram (*Gayakwad Fig. 7-17*)**")
            fig_sch = draw_sawtooth_gen_schematic(r1_kohm, r4_kohm, rp_kohm, wiper_pct, c1_uf, r2_kohm, r3_kohm, rom_kohm, vcc)
            st.pyplot(fig_sch)
            plt.close(fig_sch)
            st.caption(
                f"**DC Offset Branch:** Potentiometer $R_p = {rp_kohm:.0f}\\text{{ k}}\\Omega$ ($V_{{dc}} = {V_dc:+.2f}\\text{{ V}}$) "
                f"via $R_4 = {r4_kohm:.1f}\\text{{ k}}\\Omega$ into Pin 2 of $A_2$ ($R_{{OM}} = R_1 \\parallel R_4 = {rom_kohm:.2f}\\text{{ k}}\\Omega$)."
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
                title=f"CH1: Comparator Pulse $v_{{o1}}(t)$ (Duty Cycle = {duty_pct:.1f}%)",
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

        # -----------------------------------------------------------------
        # Secondary Tabs: Multi-Frequency Comparison (3 Frequencies) + Theory
        # -----------------------------------------------------------------
        tab_multi, tab_theory = st.tabs([
            "🔁 Repeat for Three Frequencies ($f_1, f_2, f_3$)",
            "📖 Theory & Design Procedure (Gayakwad)"
        ])

        with tab_multi:
            st.markdown(
                "Select three integrator capacitor values ($C_{1a}, C_{1b}, C_{1c}$) with fixed $R_1 = "
                f"{r1_kohm:.1f}\\text{{ k}}\\Omega$, $R_4 = {r4_kohm:.1f}\\text{{ k}}\\Omega$, and wiper $V_{{dc}} = {V_dc:+.2f}\\text{{ V}}$ "
                "to compare the sawtooth outputs across **three different frequencies** while keeping the sawtooth shape identical:"
            )

            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                c1a_uf = st.number_input("Trial 1: Capacitor $C_{1a}$ (µF)", min_value=0.005, max_value=0.500, value=0.025, step=0.005, format="%.3f")
            with fcol2:
                c1b_uf = st.number_input("Trial 2: Capacitor $C_{1b}$ (µF)", min_value=0.005, max_value=0.500, value=0.050, step=0.005, format="%.3f")
            with fcol3:
                c1c_uf = st.number_input("Trial 3: Capacitor $C_{1c}$ (µF)", min_value=0.005, max_value=0.500, value=0.100, step=0.005, format="%.3f")

            trials = [
                ("Trial 1 ($f_1$)", c1a_uf * 1e-6, "#1f77b4"),
                ("Trial 2 ($f_2$)", c1b_uf * 1e-6, "#2ca02c"),
                ("Trial 3 ($f_3$)", c1c_uf * 1e-6, "#d62728"),
            ]

            max_C1 = max(c1a_uf, c1b_uf, c1c_uf) * 1e-6
            res_max = simulate_sawtooth_generator(R1, R4, V_dc, max_C1, R2, R3, vcc, t_span=1e-3, num_pts=10)
            t_common_span = 4.0 * res_max[4]

            fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
            table_rows = []

            for idx, (label, c1_val, color) in enumerate(trials):
                t_i, _, vout_i, fo_i, T_i, t1_i, t2_i, _, _, _, vsat_i = simulate_sawtooth_generator(
                    R1, R4, V_dc, c1_val, R2, R3, vcc, t_span=t_common_span, num_pts=4000
                )
                ax = axes_cmp[idx]
                ax.plot(t_i * 1000.0, vout_i, color=color, linewidth=2.0)
                ax.set_ylim(-vsat_i * 1.2, vsat_i * 1.2)
                style_cro_axes(
                    ax,
                    title=f"{label}: $C_1 = {c1_val*1e6:.3f}$ µF, $R_1 = {r1_kohm:.1f}$ kΩ  →  $f_o = {fo_i:.1f}$ Hz ($T = {T_i*1000:.2f}$ ms)",
                    xlabel="Time (ms)" if idx == 2 else "",
                    ylabel="$v_o$ (V)"
                )
                table_rows.append({
                    "Trial": label,
                    "Capacitor C1 (µF)": f"{c1_val*1e6:.3f}",
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
                ### Theory & Design Equations (*R. A. Gayakwad, Section 7.8, Fig. 7-17*)

                1. **Triangular-to-Sawtooth Conversion via DC Offset Potentiometer ($R_p, R_4$):**
                   A triangular wave generator is converted into a sawtooth wave generator by connecting a **potentiometer $R_p$** between $+V_{{CC}}$ and $-V_{{CC}}$ and feeding its wiper voltage **$V_{{dc}}$** through resistor **$R_4$** into the inverting summing node (Pin 2) of integrator $A_2$.
                   * When the wiper is at the center ($V_{{dc}} = 0\\text{{ V}}$), the charging and discharging currents are equal ($\\pm V_{{sat}}/R_1$), producing a symmetric **triangular wave**.
                   * When $V_{{dc}} \\ne 0$, the net integration current during the two half-cycles becomes unequal:
                     $$I_{{\\text{{slow}}}} = \\frac{{V_{{sat}}}}{{R_1}} - \\frac{{|V_{{dc}}|}}{{R_4}}, \\qquad I_{{\\text{{fast}}}} = \\frac{{V_{{sat}}}}{{R_1}} + \\frac{{|V_{{dc}}|}}{{R_4}}$$

                2. **Peak-to-Peak Output Voltage ($V_{{o(pp)}}$):**
                   Determined solely by comparator $A_1$'s feedback resistors $R_2$ and $R_3$ ($R_3 > R_2$):
                   $$V_{{o(pp)}} = 2 V_p = 2 \\left(\\frac{{R_2}}{{R_3}}\\right) V_{{sat}} = {vo_pp_ideal:.2f}\\text{{ V}}$$

                3. **Ramp Time ($T_1$), Flyback Time ($T_2$), and Frequency ($f_o$):**
                   $$T_1 = \\frac{{V_{{o(pp)}} \\cdot C_1}}{{I_{{\\text{{slow}}}}}} = {T_ramp*1000:.3f}\\text{{ ms}}, \\qquad T_2 = \\frac{{V_{{o(pp)}} \\cdot C_1}}{{I_{{\\text{{fast}}}}}} = {T_flyback*1000:.3f}\\text{{ ms}}$$
                   $$T = T_1 + T_2 = {T*1000:.3f}\\text{{ ms}}, \\qquad f_o = \\frac{{1}}{{T}} = {fo:.2f}\\text{{ Hz}}$$

                4. **Offset Minimizing Resistor ($R_{{OM}}$):**
                   $$R_{{OM}} = R_1 \\parallel R_4 = {rom_kohm:.2f}\\text{{ k}}\\Omega$$
                """
            )
