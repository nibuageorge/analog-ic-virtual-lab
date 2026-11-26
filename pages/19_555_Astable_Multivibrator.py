import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 17: IC 555 Astable Multivibrator", layout="wide")
render_sidebar_header()

st.title("Experiment 17: Astable Multivibrator using IC 555 Timer")
st.markdown(
    "Design and construct an **Astable Multivibrator** using **IC 555 Timer** "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 9-21(a) IC 555 Astable Schematic
# -------------------------------------------------------------------------
def draw_555_astable_schematic(ra_kohm, rb_kohm, c_uf, c1_uf, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        # --- IC 555 Block matching Gayakwad Fig. 9-21(a) pin layout ---
        ic = (
            elm.Ic(
                pins=[
                    elm.IcPin(name='7', side='left', slot='3/3'),
                    elm.IcPin(name='6', side='left', slot='2/3'),
                    elm.IcPin(name='2', side='left', slot='1/3'),
                    elm.IcPin(name='8', side='top', slot='1/2'),
                    elm.IcPin(name='4', side='top', slot='2/2'),
                    elm.IcPin(name='1', side='bot', slot='1/2'),
                    elm.IcPin(name='5', side='bot', slot='2/2'),
                    elm.IcPin(name='3', side='right', slot='2/3'),
                ],
                pinspacing=0.65,
                edgepadH=0.45,
                edgepadW=0.55,
            )
            .label('555', loc='center', fontsize=10)
        )
        d.add(ic)

        # --- Top Supply Rail +VCC connected to Pins 8 and 4 ---
        p8_x, p8_y = ic.anchors['8']
        p4_x, p4_y = ic.anchors['4']
        top_rail_y = p8_y + 0.75

        d.add(elm.Line().up().at((p8_x, p8_y)).toy(top_rail_y))
        vcc_node = d.add(elm.Dot())
        d.add(elm.Line().up().at(vcc_node.center).length(0.55).label(f'+$V_{{CC}}$ = +{vcc:.1f} V', loc='top', fontsize=8.5))

        d.add(elm.Line().up().at((p4_x, p4_y)).toy(top_rail_y))
        d.add(elm.Line().left().tox(p8_x))

        # --- Left RC Timing Network (RA, RB, C) connected to Pins 7, 6, 2 ---
        p7_x, p7_y = ic.anchors['7']
        p6_x, p6_y = ic.anchors['6']
        p2_x, p2_y = ic.anchors['2']

        rc_x = p7_x - 1.35
        d.add(elm.Line().left().at((p8_x, top_rail_y)).tox(rc_x))

        # Resistor RA from top supply rail down to Pin 7 level
        d.add(elm.Resistor().down().at((rc_x, top_rail_y)).toy(p7_y).label(f'$R_A$\n{ra_kohm:.1f} k$\\Omega$', loc='left', fontsize=8.2))
        d7 = d.add(elm.Dot())
        d.add(elm.Line().right().at(d7.center).tox(p7_x))

        # Resistor RB from Pin 7 level down to Pin 6/2 level
        d.add(elm.Resistor().down().at(d7.center).toy(p2_y).label(f'$R_B$\n{rb_kohm:.1f} k$\\Omega$', loc='left', fontsize=8.2))
        d2 = d.add(elm.Dot())

        # Tie Pins 6 and 2 together to node d2
        j6_x = p6_x - 0.55
        d.add(elm.Line().left().at((p6_x, p6_y)).tox(j6_x))
        d.add(elm.Line().down().toy(p2_y))
        d.add(elm.Dot())
        d.add(elm.Line().right().tox(p2_x))
        d.add(elm.Line().left().at((j6_x, p2_y)).tox(rc_x))

        # Timing Capacitor C from node d2 to Ground
        d.add(elm.Capacitor().down().at(d2.center).length(1.35).label(f'$C$\n{c_uf:.3f} uF', loc='left', fontsize=8.2))
        d.add(elm.Ground())

        # --- Bottom Pins: Pin 1 (Ground) and Pin 5 (Bypass C1 = 0.01 uF) ---
        p1_x, p1_y = ic.anchors['1']
        d.add(elm.Line().down().at((p1_x, p1_y)).length(0.65))
        d.add(elm.Ground())

        p5_x, p5_y = ic.anchors['5']
        d.add(elm.Capacitor().down().at((p5_x, p5_y)).length(1.05).label(f'$C_1$\n{c1_uf:.2f} uF', loc='right', fontsize=8))
        d.add(elm.Ground())

        # --- Output Pin 3 ---
        p3_x, p3_y = ic.anchors['3']
        d.add(elm.Line().right().at((p3_x, p3_y)).length(1.1).label('Output (Pin 3)', loc='right', fontsize=8.8))

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Helper: Exact Gayakwad 555 Astable Timing & Waveform Equations
# -------------------------------------------------------------------------
def compute_555_astable(RA, RB, C, vcc):
    # Gayakwad Eqs. (9-8a), (9-8b), (9-9), (9-10), (9-11)
    tc = 0.69 * (RA + RB) * C
    td = 0.69 * RB * C
    period = tc + td
    freq = 1.0 / period
    duty_pct = (tc / period) * 100.0
    v_ut = (2.0 / 3.0) * vcc
    v_lt = (1.0 / 3.0) * vcc
    return tc, td, period, freq, duty_pct, v_ut, v_lt

def generate_555_waveforms(t_arr, tc, td, vcc):
    period = tc + td
    t_mod = np.mod(t_arr, period)
    v_ut = (2.0 / 3.0) * vcc
    v_lt = (1.0 / 3.0) * vcc

    # Output square wave: ~VCC during tc, 0 V during td
    vo = np.where(t_mod < tc, vcc, 0.0)

    # Exact exponential charging (1/3 VCC -> 2/3 VCC) and discharging (2/3 VCC -> 1/3 VCC)
    tau_c = tc / np.log(2.0)
    tau_d = td / np.log(2.0)
    vc_charge = vcc - (vcc - v_lt) * np.exp(-t_mod / tau_c)
    vc_discharge = v_ut * np.exp(-(t_mod - tc) / tau_d)
    vc = np.where(t_mod < tc, vc_charge, vc_discharge)
    return vo, vc

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Timing Resistors & Capacitor")
    ra_kohm = st.slider(
        "Resistor $R_A$ (kOhm)",
        min_value=0.5,
        max_value=100.0,
        value=2.2,
        step=0.1,
        help="Gayakwad Fig. 9-21(a) default: RA = 2.2 kOhm (must be > 0 to avoid shorting Pin 7 to +VCC)."
    )
    rb_kohm = st.slider(
        "Resistor $R_B$ (kOhm)",
        min_value=0.5,
        max_value=100.0,
        value=3.9,
        step=0.1,
        help="Gayakwad Fig. 9-21(a) default: RB = 3.9 kOhm."
    )
    c_uf = st.slider(
        "Timing Capacitor $C$ (uF)",
        min_value=0.001,
        max_value=1.000,
        value=0.100,
        step=0.005,
        format="%.3f",
        help="Gayakwad Fig. 9-21(a) default: C = 0.1 uF (gives fo = 1.45 kHz)."
    )

    st.subheader("2. Control Bypass & Supply")
    c1_uf = st.slider(
        "Bypass Capacitor $C_1$ on Pin 5 (uF)",
        min_value=0.01,
        max_value=0.10,
        value=0.01,
        step=0.01,
        format="%.2f",
        help="Gayakwad Fig. 9-21(a) default: C1 = 0.01 uF to filter noise at Pin 5."
    )
    vcc = st.slider(
        "Supply Voltage $+V_{CC}$ (V)",
        min_value=4.5,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Gayakwad Fig. 9-21(a) default: +VCC = +5 V (frequency fo is independent of VCC)."
    )

# -------------------------------------------------------------------------
# Primary Astable Calculations
# -------------------------------------------------------------------------
RA = ra_kohm * 1e3
RB = rb_kohm * 1e3
C = c_uf * 1e-6

tc, td, period, freq, duty_pct, v_ut, v_lt = compute_555_astable(RA, RB, C, vcc)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, CRO Waveforms & Repeat Tab
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Free-Running Freq ($f_o$)", format_eng(freq, "Hz"))
    m2.metric("Positive Pulse Width ($t_c$)", f"{tc*1e3:.3f} ms")
    m3.metric("Negative Pulse Width ($t_d$)", f"{td*1e3:.3f} ms")
    m4.metric("Duty Cycle ($t_c / T$)", f"{duty_pct:.1f}% (T={period*1e3:.3f} ms)")

    st.success(
        f"**IC 555 Astable Multivibrator ($f_o = {freq:.1f}$ Hz):** "
        f"Positive pulse width **$t_c = {tc*1e3:.3f}$ ms** (charging $\\frac{{1}}{{3}}V_{{CC}} \\to \\frac{{2}}{{3}}V_{{CC}}$ through $R_A + R_B$), "
        f"negative pulse width **$t_d = {td*1e3:.3f}$ ms** (discharging $\\frac{{2}}{{3}}V_{{CC}} \\to \\frac{{1}}{{3}}V_{{CC}}$ through $R_B$), "
        f"Duty Cycle **$D = {duty_pct:.1f}\\%$**."
    )

    schem_col, cro_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_555_astable_schematic(ra_kohm, rb_kohm, c_uf, c1_uf, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Thresholds:** $\\frac{{2}}{{3}}V_{{CC}} = {v_ut:.2f}$ V, $\\frac{{1}}{{3}}V_{{CC}} = {v_lt:.2f}$ V; "
            f"$f_o = \\frac{{1.45}}{{(R_A + 2R_B)C}} = {freq:.1f}$ Hz."
        )

    with cro_col:
        st.markdown("**Output & Capacitor Waveforms (Gayakwad)**")
        t_arr = np.linspace(0.0, 3.0 * period, 2500)
        vo_arr, vc_arr = generate_555_waveforms(t_arr, tc, td, vcc)
        t_ms = t_arr * 1000.0

        fig_cro, (ax_vo, ax_vc) = plt.subplots(2, 1, figsize=(7.6, 5.3), sharex=True)

        # Top Plot: Output Voltage Waveform (Pin 3)
        ax_vo.plot(t_ms, vo_arr, color="#d62728", linewidth=2.2, label=f"Output Voltage $v_o$ ($D = {duty_pct:.1f}\\%$)")
        ax_vo.axhline(vcc, color="#888888", linestyle=":", linewidth=1.0, label=f"$+V_{{CC}} = {vcc:.1f}$ V")
        ax_vo.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_vo,
            title=f"Pin 3 Output Voltage ($t_c = {tc*1e3:.3f}$ ms, $t_d = {td*1e3:.3f}$ ms, $T = {period*1e3:.3f}$ ms)",
            xlabel="",
            ylabel="Output (V)"
        )
        ax_vo.legend(loc="upper right", fontsize=8)

        # Bottom Plot: Capacitor Voltage Waveform (Pins 6 & 2)
        ax_vc.plot(t_ms, vc_arr, color="#1f77b4", linewidth=2.2, label="Capacitor Voltage $v_C(t)$")
        ax_vc.axhline(v_ut, color="#2ca02c", linestyle="--", linewidth=1.1, label=f"$\\frac{{2}}{{3}}V_{{CC}} = {v_ut:.2f}$ V")
        ax_vc.axhline(v_lt, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"$\\frac{{1}}{{3}}V_{{CC}} = {v_lt:.2f}$ V")
        ax_vc.set_ylim(0.0, 1.15 * vcc)
        style_cro_axes(
            ax_vc,
            title="Capacitor Voltage $v_C(t)$ (Charging & Discharging between 1/3 VCC and 2/3 VCC)",
            xlabel="Time (ms)",
            ylabel="Capacitor (V)"
        )
        ax_vc.legend(loc="upper right", fontsize=8)

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
            f"Compare **Trial 1** (current frequency **$f_{{o1}} = {freq:.1f}$ Hz**, $R_A = {ra_kohm:.1f}$ kOhm, "
            f"$R_B = {rb_kohm:.1f}$ kOhm, $C = {c_uf:.3f}$ uF) with a **second frequency ($f_{{o2}}$)** by choosing "
            f"new component values below:"
        )

        col_r1, col_r2, col_r3 = st.columns(3, gap="medium")
        with col_r1:
            ra2_kohm = st.slider("Trial 2 Resistor $R_{A2}$ (kOhm)", 0.5, 100.0, ra_kohm, 0.1)
        with col_r2:
            rb2_kohm = st.slider("Trial 2 Resistor $R_{B2}$ (kOhm)", 0.5, 100.0, rb_kohm, 0.1)
        with col_r3:
            c2_uf = st.slider(
                "Trial 2 Capacitor $C_2$ (uF)",
                min_value=0.005,
                max_value=1.000,
                value=round(min(c_uf * 2.0, 1.0), 3) if c_uf < 0.4 else round(c_uf / 2.0, 3),
                step=0.005,
                format="%.3f"
            )

        tc2, td2, period2, freq2, duty2, _, _ = compute_555_astable(ra2_kohm * 1e3, rb2_kohm * 1e3, c2_uf * 1e-6, vcc)

        t_cmp_max = max(period, period2) * 2.5
        t_cmp = np.linspace(0.0, t_cmp_max, 2500)
        t_cmp_ms = t_cmp * 1000.0

        vo1_c, vc1_c = generate_555_waveforms(t_cmp, tc, td, vcc)
        vo2_c, vc2_c = generate_555_waveforms(t_cmp, tc2, td2, vcc)

        fig_rep, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(8.8, 4.8), sharex=True)

        ax_t1.plot(t_cmp_ms, vo1_c, color="#d62728", linewidth=2.0, label="Output $v_{o1}(t)$")
        ax_t1.plot(t_cmp_ms, vc1_c, color="#1f77b4", linestyle="--", linewidth=1.4, label="Capacitor $v_{C1}(t)$")
        ax_t1.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_t1,
            title=f"Trial 1: fo1 = {freq:.1f} Hz (tc = {tc*1e3:.3f} ms, td = {td*1e3:.3f} ms, D = {duty_pct:.1f}%)",
            xlabel="",
            ylabel="Voltage (V)"
        )
        ax_t1.legend(loc="upper right", fontsize=8)

        ax_t2.plot(t_cmp_ms, vo2_c, color="#2ca02c", linewidth=2.0, label="Output $v_{o2}(t)$")
        ax_t2.plot(t_cmp_ms, vc2_c, color="#ff7f0e", linestyle="--", linewidth=1.4, label="Capacitor $v_{C2}(t)$")
        ax_t2.set_ylim(-0.15 * vcc, 1.25 * vcc)
        style_cro_axes(
            ax_t2,
            title=f"Trial 2: fo2 = {freq2:.1f} Hz (tc = {tc2*1e3:.3f} ms, td = {td2*1e3:.3f} ms, D = {duty2:.1f}%)",
            xlabel="Time (ms)",
            ylabel="Voltage (V)"
        )
        ax_t2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_rep)
        plt.close(fig_rep)

        st.markdown("**Observation Table Comparing Two Free-Running Frequencies**")
        st.table([
            {
                "Trial": "Trial 1 (Initial)",
                "RA (kOhm)": f"{ra_kohm:.2f}",
                "RB (kOhm)": f"{rb_kohm:.2f}",
                "C (uF)": f"{c_uf:.3f}",
                "Positive Width tc (ms)": f"{tc*1e3:.3f}",
                "Negative Width td (ms)": f"{td*1e3:.3f}",
                "Period T (ms)": f"{period*1e3:.3f}",
                "Frequency fo (Hz)": f"{freq:.1f}",
                "Duty Cycle (%)": f"{duty_pct:.1f}%"
            },
            {
                "Trial": "Trial 2 (Repeat)",
                "RA (kOhm)": f"{ra2_kohm:.2f}",
                "RB (kOhm)": f"{rb2_kohm:.2f}",
                "C (uF)": f"{c2_uf:.3f}",
                "Positive Width tc (ms)": f"{tc2*1e3:.3f}",
                "Negative Width td (ms)": f"{td2*1e3:.3f}",
                "Period T (ms)": f"{period2*1e3:.3f}",
                "Frequency fo (Hz)": f"{freq2:.1f}",
                "Duty Cycle (%)": f"{duty2:.1f}%"
            }
        ])

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Design Equations (R. A. Gayakwad)

            1. **Circuit Operation:**
               * Initially, when the output (Pin 3) is HIGH ($\\cong V_{{CC}}$), external capacitor $C$ charges toward $+V_{{CC}}$ through **$R_A + R_B$**[cite: 7, 8].
               * When capacitor voltage $v_C$ reaches **$\\frac{{2}}{{3}}V_{{CC}}$**, internal Comparator 1 triggers the flip-flop, switching the output LOW ($0\\text{{ V}}$) and turning ON internal discharge transistor $Q_1$ (Pin 7)[cite: 7].
               * Capacitor $C$ now discharges toward $0\\text{{ V}}$ solely through **$R_B$** and transistor $Q_1$ until $v_C$ drops to **$\\frac{{1}}{{3}}V_{{CC}}$**, where Comparator 2 sets the flip-flop HIGH again and the cycle repeats[cite: 7].

            2. **Pulse Width & Frequency Equations (*Gayakwad Eqs. 9-8 to 9-11*):**
               * **Positive Pulse Width (Charging Time $t_c$):**
                 $$t_c = 0.69 (R_A + R_B) C = {tc*1e3:.3f}\\text{{ ms}} \\quad \\text{{(Eq. 9-8a)}}$$
               * **Negative Pulse Width (Discharging Time $t_d$):**
                 $$t_d = 0.69 R_B C = {td*1e3:.3f}\\text{{ ms}} \\quad \\text{{(Eq. 9-8b)}}$$
               * **Total Period ($T$) and Free-Running Frequency ($f_o$):**
                 $$T = t_c + t_d = 0.69 (R_A + 2 R_B) C = {period*1e3:.3f}\\text{{ ms}} \\quad \\text{{(Eq. 9-9)}}$$
                 $$f_o = \\frac{{1}}{{T}} = \\frac{{1.45}}{{(R_A + 2 R_B) C}} = {freq:.1f}\\text{{ Hz}} \\quad \\text{{(Eq. 9-10)}}$$
               * **Duty Cycle:**
                 $$\\%\\text{{ duty cycle}} = \\frac{{t_c}}{{T}} \\times 100 = \\frac{{R_A + R_B}}{{R_A + 2 R_B}} (100) = {duty_pct:.1f}\\% \\quad \\text{{(Eq. 9-11)}}$$
                 *(Note: Because $R_A > 0$ is required to prevent shorting Pin 7 directly to $+V_{{CC}}$ when $Q_1$ conducts, the standard circuit produces duty cycles strictly greater than $50\\%$[cite: 8].)*
            """
        )
