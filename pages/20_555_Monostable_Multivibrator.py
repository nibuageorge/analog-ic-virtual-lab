import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 18: IC 555 Monostable Multivibrator", layout="wide")
render_sidebar_header()

st.title("Experiment 18: Monostable Multivibrator using IC 555 Timer")
st.markdown(
    "Design and construct a **Monostable Multivibrator** using **IC 555 Timer**, "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 9-16(a) IC 555 Monostable Schematic
# -------------------------------------------------------------------------
def draw_555_monostable_schematic(ra_kohm, c_uf, c1_uf, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        # --- IC 555 Block matching Gayakwad Fig. 9-16(a) pinout ---
        ic = (
            elm.Ic(
                pins=[
                    elm.IcPin(name='2', side='left', slot='2/3'),
                    elm.IcPin(name='3', side='left', slot='1/3'),
                    elm.IcPin(name='8', side='top', slot='1/2'),
                    elm.IcPin(name='4', side='top', slot='2/2'),
                    elm.IcPin(name='5', side='bot', slot='1/2'),
                    elm.IcPin(name='7', side='right', slot='3/3'),
                    elm.IcPin(name='6', side='right', slot='2/3'),
                    elm.IcPin(name='1', side='right', slot='1/3'),
                ],
                pinspacing=0.65,
                edgepadH=0.45,
                edgepadW=0.55,
            )
            .label('555', loc='center', fontsize=10)
        )
        d.add(ic)

        # --- Left Side: Pin 2 (Trigger Input) and Pin 3 (Output) ---
        p2_x, p2_y = ic.anchors['2']
        d.add(elm.Line().left().at((p2_x, p2_y)).length(1.25).label('Trigger Input', loc='top', fontsize=8.2))

        p3_x, p3_y = ic.anchors['3']
        d.add(elm.Line().left().at((p3_x, p3_y)).length(1.25).label('Output', loc='bottom', fontsize=8.5))

        # --- Top Supply Rail +VCC connected to Pins 8 and 4 (Reset) ---
        p8_x, p8_y = ic.anchors['8']
        p4_x, p4_y = ic.anchors['4']
        top_rail_y = p8_y + 0.75

        d.add(elm.Line().up().at((p8_x, p8_y)).toy(top_rail_y))
        d.add(elm.Line().left().length(0.55).label(f'+$V_{{CC}}$ = +{vcc:.1f} V', loc='left', fontsize=8.5))

        d.add(elm.Line().up().at((p4_x, p4_y)).toy(top_rail_y).label('Reset', loc='right', fontsize=7.5))
        d.add(elm.Dot())
        d.add(elm.Line().left().tox(p8_x))

        # --- Right Side: RC Timing Network (RA, C) on Pins 7, 6 & Ground on Pin 1 ---
        p7_x, p7_y = ic.anchors['7']
        p6_x, p6_y = ic.anchors['6']
        p1_x, p1_y = ic.anchors['1']

        rc_x = p7_x + 1.15
        # Extend top +VCC rail right to rc_x
        d.add(elm.Line().right().at((p4_x, top_rail_y)).tox(rc_x))

        # Resistor RA from +VCC rail down to Pin 7 level
        d.add(elm.Resistor().down().at((rc_x, top_rail_y)).toy(p7_y).label(f'$R_A$\n{ra_kohm:.1f} k$\\Omega$', loc='right', fontsize=8.2))
        n7 = d.add(elm.Dot())
        d.add(elm.Line().left().at(n7.center).tox(p7_x))

        # Tie Pin 6 to Pin 7 node
        j6_x = p6_x + 0.45
        d.add(elm.Line().right().at((p6_x, p6_y)).tox(j6_x))
        d.add(elm.Line().up().toy(p7_y))
        d.add(elm.Dot())

        # Capacitor C from Pin 7 node down to Pin 1 Ground level
        d.add(elm.Capacitor().down().at(n7.center).toy(p1_y).label(f'$C$\n{c_uf:.3f} uF', loc='right', fontsize=8.2))
        gnd_node = d.add(elm.Dot())
        d.add(elm.Line().left().at(gnd_node.center).tox(p1_x))
        d.add(elm.Line().down().at(gnd_node.center).length(0.55))
        d.add(elm.Ground())

        # --- Bottom Pin 5 Bypass Capacitor C1 = 0.01 uF to Ground ---
        p5_x, p5_y = ic.anchors['5']
        d.add(elm.Line().down().at((p5_x, p5_y)).length(0.55))
        d.add(elm.Capacitor().right().tox(p1_x).label(f'$C_1$ = {c1_uf:.2f} uF', loc='bottom', fontsize=7.8))
        d.add(elm.Line().up().toy(p1_y))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Helper: Monostable Timing & Waveform Generator (Gayakwad Fig. 9-16c)
# -------------------------------------------------------------------------
def simulate_555_monostable(t_arr, tp, f_trig, t_trig_width, v_trig_low, vcc):
    T_trig = 1.0 / f_trig
    v_th_trig = (1.0 / 3.0) * vcc
    v_ut = (2.0 / 3.0) * vcc

    # Negative-going trigger pulses resting at +VCC and dropping to v_trig_low
    t_mod = np.mod(t_arr, T_trig)
    # Offset first trigger slightly (at 0.12 * T_trig) so initial low state is visible
    t_shifted = np.mod(t_arr - 0.12 * T_trig, T_trig)
    v_trig = np.where((t_arr >= 0.12 * T_trig) & (t_shifted < t_trig_width), v_trig_low, vcc)

    vo = np.zeros_like(t_arr)
    vc = np.zeros_like(t_arr)

    if v_trig_low >= v_th_trig:
        # Trigger never drops below 1/3 VCC -> output stays LOW (0 V)
        return v_trig, vo, vc, False

    tau = tp / np.log(3.0)  # Since 1 - exp(-tp/tau) = 2/3 -> tp = tau * ln(3) ≈ 1.1 * RA * C
    in_pulse = False
    t_start = 0.0
    prev_trig_high = True

    for i, t_now in enumerate(t_arr):
        trig_low_now = v_trig[i] < v_th_trig
        falling_edge = prev_trig_high and trig_low_now
        prev_trig_high = not trig_low_now

        if not in_pulse and falling_edge:
            in_pulse = True
            t_start = t_now

        if in_pulse:
            dt = t_now - t_start
            if dt < tp:
                vo[i] = vcc
                vc[i] = vcc * (1.0 - np.exp(-dt / tau))
            else:
                # Reached 2/3 VCC -> output switches LOW and capacitor discharges rapidly
                if not trig_low_now:
                    in_pulse = False
                    vo[i] = 0.0
                    vc[i] = 0.0
                else:
                    # If trigger is still held LOW past tp, output stays HIGH until trigger rises
                    vo[i] = vcc
                    vc[i] = v_ut
        else:
            vo[i] = 0.0
            vc[i] = 0.0

    return v_trig, vo, vc, True

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Monostable Timing ($R_A, C$)")
    ra_kohm = st.slider(
        "Timing Resistor $R_A$ (kOhm)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Sets output pulse width tp = 1.1 * RA * C (default 10 kOhm with 0.1 uF -> tp = 1.10 ms)."
    )
    c_uf = st.slider(
        "Timing Capacitor $C$ (uF)",
        min_value=0.005,
        max_value=1.000,
        value=0.100,
        step=0.001,
        format="%.3f",
        help="Default 0.100 uF gives tp = 1.10 ms."
    )

    st.subheader("2. Trigger Input Signal (Pin 2)")
    f_trig = st.slider(
        "Trigger Pulse Frequency $f_{trig}$ (Hz)",
        min_value=50,
        max_value=2000,
        value=400,
        step=25,
        help="Trigger period T_trig = 1/f_trig should be larger than output pulse width tp."
    )
    t_trig_ms = st.slider(
        "Trigger Pulse Width $t_{trig}$ (ms)",
        min_value=0.05,
        max_value=2.00,
        value=0.20,
        step=0.05,
        help="Gayakwad notes trigger pulse width must be smaller than expected output pulse width tp."
    )
    v_trig_low = st.slider(
        "Trigger Low Level $V_{trig(low)}$ (V)",
        min_value=0.0,
        max_value=4.0,
        value=0.5,
        step=0.1,
        help="Must drop below 1/3 VCC to trigger the internal lower comparator."
    )

    st.subheader("3. Bypass Capacitor & Supply")
    c1_uf = st.slider("Bypass Capacitor $C_1$ on Pin 5 (uF)", 0.01, 0.10, 0.01, 0.01, format="%.2f")
    vcc = st.slider("Supply Voltage $+V_{CC}$ (V)", 4.5, 15.0, 5.0, 0.5)

# -------------------------------------------------------------------------
# Primary Monostable Calculations (Gayakwad Eq. 9-2)
# -------------------------------------------------------------------------
RA = ra_kohm * 1e3
C = c_uf * 1e-6
tp = 1.1 * RA * C
T_trig = 1.0 / f_trig
t_trig_sec = t_trig_ms * 1e-3
v_lt = (1.0 / 3.0) * vcc
v_ut = (2.0 / 3.0) * vcc
duty_out_pct = min((tp / T_trig) * 100.0, 100.0)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, Waveforms & Repeat Tab
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Output Pulse Width ($t_p$)", f"{tp*1e3:.2f} ms")
    m2.metric("Trigger Frequency ($f_{trig}$)", format_eng(f_trig, "Hz"))
    m3.metric("Output Duty Cycle ($t_p / T_{trig}$)", f"{duty_out_pct:.1f}%")
    m4.metric("Thresholds ($\\frac{1}{3}V_{CC}, \\frac{2}{3}V_{CC}$)", f"{v_lt:.2f} V, {v_ut:.2f} V")

    if v_trig_low >= v_lt:
        st.error(
            f"**Not Triggered ($V_{{trig(low)}} = {v_trig_low:.2f}$ V $\\ge \\frac{{1}}{{3}}V_{{CC}} = {v_lt:.2f}$ V):** "
            f"Gayakwad requires a negative-going trigger pulse dropping below **$\\frac{{1}}{{3}}V_{{CC}}$** (amplitude $> \\frac{{1}}{{3}}V_{{CC}}$). "
            f"Lower $V_{{trig(low)}}$ below **{v_lt:.2f} V**."
        )
    elif t_trig_sec >= tp or tp >= T_trig:
        st.warning(
            f"**Pulse Width / Period Overlap ($t_p = {tp*1e3:.2f}$ ms, $t_{{trig}} = {t_trig_ms:.2f}$ ms, $T_{{trig}} = {T_trig*1e3:.2f}$ ms):** "
            f"For clean monostable operation, keep **$t_{{trig}} < t_p < T_{{trig}}$**."
        )
    else:
        st.success(
            f"**Monostable Pulse Generated ($t_p = 1.1 R_A C = {tp*1e3:.2f}$ ms):** "
            f"Each negative trigger pulse ($f_{{trig}} = {f_trig}$ Hz, $T_{{trig}} = {T_trig*1e3:.2f}$ ms) produces a single "
            f"clean HIGH pulse of width **$t_p = {tp*1e3:.2f}$ ms** as $C$ charges from $0$ V to $\\frac{{2}}{{3}}V_{{CC}} = {v_ut:.2f}$ V."
        )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_555_monostable_schematic(ra_kohm, c_uf, c1_uf, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Gayakwad Eq. (9-2):** $t_p = 1.1 R_A C = 1.1({ra_kohm:.1f}\\text{{ k}}\\Omega)({c_uf:.3f}\\text{{ uF}}) = {tp*1e3:.2f}\\text{{ ms}}$."
        )

    with cro_col:
        st.markdown("**Trigger, Output & Capacitor Waveforms (*Gayakwad Fig. 9-16c*)**")
        t_sim = np.linspace(0.0, 3.0 * T_trig, 3000)
        v_trig_arr, vo_arr, vc_arr, triggered = simulate_555_monostable(
            t_sim, tp, f_trig, t_trig_sec, v_trig_low, vcc
        )
        t_ms = t_sim * 1000.0

        fig_cro, (ax_tr, ax_vo, ax_vc) = plt.subplots(3, 1, figsize=(7.6, 5.6), sharex=True)

        # Trace 1: Trigger Input (Pin 2)
        ax_tr.plot(t_ms, v_trig_arr, color="#2ca02c", linewidth=2.0, label="Trigger Input (Pin 2)")
        ax_tr.axhline(v_lt, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"$\\frac{{1}}{{3}}V_{{CC}} = {v_lt:.2f}$ V")
        ax_tr.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_tr,
            title=f"CH1: Negative-Going Trigger Input ($f_{{trig}} = {f_trig}$ Hz, $t_{{trig}} = {t_trig_ms:.2f}$ ms)",
            xlabel="",
            ylabel="Trigger (V)"
        )
        ax_tr.legend(loc="upper right", fontsize=7.5)

        # Trace 2: Output Pulse (Pin 3)
        ax_vo.plot(t_ms, vo_arr, color="#d62728", linewidth=2.1, label=f"Output Pulse $v_o$ ($t_p = {tp*1e3:.2f}$ ms)")
        ax_vo.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_vo,
            title=f"CH2: Monostable Output Pulse $v_o(t)$ ($t_p = 1.1 R_A C = {tp*1e3:.2f}$ ms)",
            xlabel="",
            ylabel="Output (V)"
        )
        ax_vo.legend(loc="upper right", fontsize=7.5)

        # Trace 3: Capacitor Voltage (Pins 6 & 2)
        ax_vc.plot(t_ms, vc_arr, color="#1f77b4", linewidth=2.1, label="Capacitor Voltage $v_C(t)$")
        ax_vc.axhline(v_ut, color="#d62728", linestyle="--", linewidth=1.0, label=f"$\\frac{{2}}{{3}}V_{{CC}} = {v_ut:.2f}$ V")
        ax_vc.set_ylim(-0.1 * vcc, 1.15 * vcc)
        style_cro_axes(
            ax_vc,
            title="CH3: Capacitor Voltage $v_C(t)$ (Charges $0 \\to \\frac{2}{3}V_{CC}$, then discharges via $Q_1$)",
            xlabel="Time (ms)",
            ylabel="$v_C$ (V)"
        )
        ax_vc.legend(loc="upper right", fontsize=7.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    # ---------------------------------------------------------------------
    # Secondary Tabs: Repeat Experiment for One More Frequency + Theory
    # ---------------------------------------------------------------------
    tab_repeat, tab_theory = st.tabs([
        "Repeat Experiment for One More Frequency",
        "Theory & Design Equations (Gayakwad)"
    ])

    with tab_repeat:
        st.markdown(
            f"Compare **Trial 1** (current trigger frequency **$f_{{trig1}} = {f_trig}$ Hz**, pulse width **$t_{{p1}} = {tp*1e3:.2f}$ ms**) "
            f"with a **second trigger frequency ($f_{{trig2}}$)** and/or second pulse width ($t_{{p2}}$):"
        )

        col_r1, col_r2, col_r3 = st.columns(3, gap="medium")
        with col_r1:
            f_trig2 = st.slider(
                "Trial 2 Trigger Frequency $f_{trig2}$ (Hz)",
                min_value=50,
                max_value=2000,
                value=600 if f_trig != 600 else 300,
                step=25
            )
        with col_r2:
            ra2_kohm = st.slider("Trial 2 Resistor $R_{A2}$ (kOhm)", 1.0, 100.0, ra_kohm, 0.5)
        with col_r3:
            c2_uf = st.slider("Trial 2 Capacitor $C_2$ (uF)", 0.005, 1.000, c_uf, 0.005, format="%.3f")

        tp2 = 1.1 * (ra2_kohm * 1e3) * (c2_uf * 1e-6)
        T_trig2 = 1.0 / f_trig2
        duty2_pct = min((tp2 / T_trig2) * 100.0, 100.0)

        t_cmp_max = max(T_trig, T_trig2) * 3.0
        t_cmp = np.linspace(0.0, t_cmp_max, 3000)
        t_cmp_ms = t_cmp * 1000.0

        vtr1, vo1_c, vc1_c, _ = simulate_555_monostable(t_cmp, tp, f_trig, t_trig_sec, v_trig_low, vcc)
        vtr2, vo2_c, vc2_c, _ = simulate_555_monostable(t_cmp, tp2, f_trig2, t_trig_sec, v_trig_low, vcc)

        fig_rep, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(8.8, 4.9), sharex=True)

        ax_t1.plot(t_cmp_ms, vtr1, color="#2ca02c", linestyle=":", linewidth=1.4, label="Trigger 1")
        ax_t1.plot(t_cmp_ms, vo1_c, color="#d62728", linewidth=2.0, label="Output $v_{o1}(t)$")
        ax_t1.plot(t_cmp_ms, vc1_c, color="#1f77b4", linestyle="--", linewidth=1.4, label="Capacitor $v_{C1}(t)$")
        ax_t1.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_t1,
            title=f"Trial 1: f_trig1 = {f_trig} Hz (T_trig1 = {T_trig*1e3:.2f} ms, Output Pulse tp1 = {tp*1e3:.2f} ms)",
            xlabel="",
            ylabel="Voltage (V)"
        )
        ax_t1.legend(loc="upper right", fontsize=7.8)

        ax_t2.plot(t_cmp_ms, vtr2, color="#2ca02c", linestyle=":", linewidth=1.4, label="Trigger 2")
        ax_t2.plot(t_cmp_ms, vo2_c, color="#9467bd", linewidth=2.0, label="Output $v_{o2}(t)$")
        ax_t2.plot(t_cmp_ms, vc2_c, color="#ff7f0e", linestyle="--", linewidth=1.4, label="Capacitor $v_{C2}(t)$")
        ax_t2.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_t2,
            title=f"Trial 2: f_trig2 = {f_trig2} Hz (T_trig2 = {T_trig2*1e3:.2f} ms, Output Pulse tp2 = {tp2*1e3:.2f} ms)",
            xlabel="Time (ms)",
            ylabel="Voltage (V)"
        )
        ax_t2.legend(loc="upper right", fontsize=7.8)

        plt.tight_layout()
        st.pyplot(fig_rep)
        plt.close(fig_rep)

        st.markdown("**Observation Table Comparing Two Frequencies**")
        st.table([
            {
                "Trial": "Trial 1 (Initial)",
                "Trigger Freq f_trig (Hz)": f"{f_trig}",
                "Trigger Period T_trig (ms)": f"{T_trig*1e3:.2f}",
                "RA (kOhm)": f"{ra_kohm:.1f}",
                "C (uF)": f"{c_uf:.3f}",
                "Theoretical tp = 1.1*RA*C (ms)": f"{tp*1e3:.2f}",
                "Output Duty Cycle (%)": f"{duty_out_pct:.1f}%"
            },
            {
                "Trial": "Trial 2 (Repeat)",
                "Trigger Freq f_trig (Hz)": f"{f_trig2}",
                "Trigger Period T_trig (ms)": f"{T_trig2*1e3:.2f}",
                "RA (kOhm)": f"{ra2_kohm:.1f}",
                "C (uF)": f"{c2_uf:.3f}",
                "Theoretical tp = 1.1*RA*C (ms)": f"{tp2*1e3:.2f}",
                "Output Duty Cycle (%)": f"{duty2_pct:.1f}%"
            }
        ])

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (R. A. Gayakwad)

            1. **Monostable Operation:**
               * In the stable state (before triggering), internal discharge transistor $Q_1$ is ON, clamping capacitor $C$ at $0\\text{{ V}}$, and the output (Pin 3) is LOW ($0\\text{{ V}}$)[cite: 10].
               * When a negative-going trigger pulse applied to Pin 2 drops below **$\\frac{{1}}{{3}}V_{{CC}}$**, Comparator 2 sets the internal flip-flop, turning $Q_1$ OFF and driving the output (Pin 3) HIGH ($\\cong V_{{CC}}$)[cite: 10].
               * Capacitor $C$ now charges exponentially toward $+V_{{CC}}$ through **$R_A$** until $v_C$ reaches **$\\frac{{2}}{{3}}V_{{CC}}$**, at which point Comparator 1 resets the flip-flop, switches the output back to LOW, and rapidly discharges $C$ through transistor $Q_1$[cite: 10].

            2. **Output Pulse Width Equation (*Gayakwad Eq. 9-2*):**
               $$t_p = 1.1 R_A C = 1.1 ({ra_kohm:.1f}\\text{{ k}}\\Omega)({c_uf:.3f}\\text{{ uF}}) = {tp*1e3:.2f}\\text{{ ms}}$$
               * The output pulse width $t_p$ is independent of the supply voltage $V_{{CC}}$ and depends solely on $R_A$ and $C$.
               * **Trigger Conditions:** The trigger pulse width must be smaller than $t_p$, and its negative-going amplitude must exceed $\\frac{{1}}{{3}}V_{{CC}}$ (dropping below $\\frac{{1}}{{3}}V_{{CC}}$)[cite: 10].
            """
        )
