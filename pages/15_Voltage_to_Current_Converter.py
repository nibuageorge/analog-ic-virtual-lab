import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 15: Voltage-to-Current Converter", layout="wide")
render_sidebar_header()

st.title("Experiment 15: Voltage-to-Current Converter with Grounded Load ($\\mu$A741)")
st.markdown(
    "Design and construct a **Voltage-to-Current (V-to-I) Converter with Grounded Load** using **$\\mu$A741**, "
    " (Ref. Gayakwad)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 6-19 V-to-I Converter Schematic
# -------------------------------------------------------------------------
def draw_v_to_i_schematic(r_kohm, rl_kohm, vin_dc, il_ma, vcc):
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.2)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Output Node Vo = 2*V1 ---
        d.add(elm.Line().right().at(op.out).length(0.85))
        out_tap = d.add(elm.Dot())
        ox, oy = out_tap.center
        d.add(elm.Line().right().at((ox, oy)).length(0.75).label('$V_o = 2V_1$', loc='right', fontsize=8.8))

        # --- Upper Negative Feedback Path (R to Ground & R to Output) ---
        inv_x, inv_y = op.in1
        top_y = inv_y + 1.25
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.45).label('$I_B \\cong 0$', loc='bottom', fontsize=7.5))
        d.add(elm.Line().up().toy(top_y))
        v2_node = d.add(elm.Dot().label('$V_2$', loc='top', fontsize=8.5))
        v2_x, v2_y = v2_node.center

        # Upper Left Resistor R to Ground
        d.add(elm.Resistor().left().at((v2_x, v2_y)).length(2.05).label(f'$R$\n{r_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
        d.add(elm.Line().down().length(0.35))
        d.add(elm.Ground())

        # Upper Right Resistor R to Output Vo
        d.add(elm.Resistor().right().at((v2_x, v2_y)).tox(ox).label(f'$R$\n{r_kohm:.1f} k$\\Omega$', loc='top', fontsize=8.2))
        d.add(elm.Line().down().toy(oy))

        # --- Lower Positive Feedback Bridge (R from Vin, R from Vo, Load to GND) ---
        noninv_x, noninv_y = op.in2
        bot_y = noninv_y - 1.25
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.45).label('$I_B \\cong 0$', loc='top', fontsize=7.5))
        d.add(elm.Line().down().toy(bot_y))
        v1_node = d.add(elm.Dot().label('$V_1$', loc='top', fontsize=8.5))
        v1_x, v1_y = v1_node.center

        # Lower Left Resistor R from Input Voltage Vin
        d.add(elm.Resistor().left().at((v1_x, v1_y)).length(2.05).label(f'$R$ ($I_1$)\n{r_kohm:.1f} k$\\Omega$', loc='top', fontsize=8))
        src_x = d.here[0]
        d.add(elm.Line().down().at((src_x, v1_y)).length(0.25))
        d.add(elm.SourceV().down().length(1.15).label(f'$V_{{in}}$\n{vin_dc:+.1f} V', loc='left', fontsize=8.2))
        d.add(elm.Ground())

        # Lower Right Resistor R from Output Vo to Node V1
        d.add(elm.Resistor().right().at((v1_x, v1_y)).tox(ox).label(f'$R$ ($I_2$)\n{r_kohm:.1f} k$\\Omega$', loc='top', fontsize=8))
        d.add(elm.Line().up().toy(oy))

        # Grounded Load Resistor RL below Node V1
        d.add(elm.Resistor().down().at((v1_x, v1_y)).length(1.4).label(f'Load $R_L$ = {rl_kohm:.1f} k$\\Omega$\n($I_L$ = {il_ma:+.2f} mA)', loc='bottom', fontsize=8))
        d.add(elm.Ground())

        return d.draw(show=False).fig

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Plots)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Bridge & Grounded Load Resistors")
    r_kohm = st.slider(
        "Equal Bridge Resistors $R$ (kOhm)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
        help="All four bridge resistors R must be equal so IL = Vin / R (default 10 kOhm -> 0.10 mA/V)."
    )
    rl_kohm = st.slider(
        "Grounded Load Resistance $R_L$ (kOhm)",
        min_value=0.1,
        max_value=25.0,
        value=2.0,
        step=0.1,
        help="Gayakwad notes circuit performs linearly provided load size RL <= R."
    )

    st.subheader("2. Input Voltage Signal ($V_{in}$)")
    vin_dc = st.slider(
        "DC Input Voltage $V_{in}$ (V)",
        min_value=-10.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Sets grounded load current IL = Vin / R independent of RL (for RL <= R)."
    )
    vin_ac_peak = st.slider(
        "AC Peak Input $V_p$ for CRO (V)",
        min_value=0.5,
        max_value=10.0,
        value=4.0,
        step=0.5
    )
    freq_hz = st.slider(
        "AC Test Frequency $f$ (Hz)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

    st.subheader("3. Power Supply")
    vcc = st.slider("Supply Voltage $\\pm V_{CC} / \\pm V_{EE}$ (V)", 5.0, 18.0, 15.0, 1.0)

# -------------------------------------------------------------------------
# Circuit Calculations (Gayakwad Section 6-9, Eqs. 6-21a & 6-21b)
# -------------------------------------------------------------------------
R = r_kohm * 1e3
RL = rl_kohm * 1e3
vsat = max(vcc - 1.0, 0.5)

# Ideal load current IL = Vin / R, load voltage V1 = IL * RL, op-amp output Vo = 2 * V1
il_ideal_amp = vin_dc / R
vo_ideal = 2.0 * il_ideal_amp * RL

# If |Vo_ideal| exceeds Vsat, op-amp saturates at +/- Vsat and V1 = (Vin + Vsat) / (2 + R/RL)
if abs(vo_ideal) > vsat:
    vo_actual = float(np.sign(vo_ideal) * vsat)
    v1_actual = (vin_dc + vo_actual) / (2.0 + (R / RL))
    il_actual_amp = v1_actual / RL
    saturated = True
else:
    vo_actual = float(vo_ideal)
    v1_actual = float(il_ideal_amp * RL)
    il_actual_amp = float(il_ideal_amp)
    saturated = False

il_actual_ma = il_actual_amp * 1e3
gm_ma_per_v = 1.0 / r_kohm
rl_max_kohm = (vsat * r_kohm) / (2.0 * max(abs(vin_dc), 1e-6))

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, Transfer Curves & CRO Waveforms
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Load Current ($I_L = V_{in}/R$)", f"{il_actual_ma:+.3f} mA")
    m2.metric("Transconductance ($I_L / V_{in}$)", f"{gm_ma_per_v:.3f} mA/V")
    m3.metric("Load Voltage ($V_1 = I_L R_L$)", f"{v1_actual:+.2f} V")
    m4.metric("Op-Amp Output ($V_o = 2 V_1$)", f"{vo_actual:+.2f} V")

    if saturated:
        st.warning(
            f"**Op-Amp Output Saturated ($|V_{{o(ideal)}}| = {abs(vo_ideal):.2f}$ V > $V_{{sat}} = {vsat:.1f}$ V):** "
            f"Because $R_L$ is too large for $V_{{in}} = {vin_dc:+.1f}$ V, $V_o$ clipped at $\\pm {vsat:.1f}$ V. "
            f"Keep **$R_L \\le {min(r_kohm, rl_max_kohm):.1f}$ kOhm** (Gayakwad rule: load size $R_L \\le R$) for constant-current operation."
        )
    elif rl_kohm > r_kohm:
        st.info(
            f"**Load Notice ($R_L = {rl_kohm:.1f}$ kOhm > $R = {r_kohm:.1f}$ kOhm):** "
            f"Current $I_L = {il_actual_ma:+.3f}$ mA is still linear here, but Gayakwad recommends **$R_L \\le R$** "
            f"so $V_o = 2 I_L R_L$ does not saturate at higher input voltages."
        )
    else:
        st.success(
            f"**Constant Grounded-Load Current ($I_L = V_{{in}} / R$):** "
            f"Input **$V_{{in}} = {vin_dc:+.1f}$ V** with **$R = {r_kohm:.1f}$ kOhm** drives **$I_L = {il_actual_ma:+.3f}$ mA** "
            f"through grounded load **$R_L = {rl_kohm:.1f}$ kOhm** ($V_o = 2 V_1 = {vo_actual:+.2f}$ V)."
        )

    schem_col, plot_col = st.columns([1.08, 1.42], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (Gayakwad)**")
        fig_sch = draw_v_to_i_schematic(r_kohm, rl_kohm, vin_dc, il_actual_ma, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Gayakwad Eq. (6-21b):** $I_L = V_{{in}} / R$; $V_o = 2 V_1$; "
            f"Valid for grounded load $R_L \\le R = {r_kohm:.1f}\\text{{ k}}\\Omega$."
        )

    with plot_col:
        st.markdown("**V-to-I Transfer & Load Regulation Curves**")
        fig_tr, (ax_vi, ax_rl) = plt.subplots(2, 1, figsize=(7.6, 5.3))

        # Top Plot: Load Current IL vs. Input Voltage Vin
        vin_sw = np.linspace(-10.0, 10.0, 400)
        il_id_sw = (vin_sw / R) * 1e3
        vo_id_sw = 2.0 * (vin_sw / R) * RL
        il_act_sw = np.where(
            np.abs(vo_id_sw) > vsat,
            ((vin_sw + np.sign(vo_id_sw) * vsat) / (2.0 * RL + R)) * 1e3,
            il_id_sw
        )
        ax_vi.plot(vin_sw, il_act_sw, color="#1f77b4", linewidth=2.3, label=f"Load Current $I_L$ ($R_L = {rl_kohm:.1f}$ kOhm)")
        ax_vi.plot(vin_sw, il_id_sw, color="#888888", linestyle=":", linewidth=1.1, label="Ideal $I_L = V_{in}/R$")
        ax_vi.scatter([vin_dc], [il_actual_ma], color="#d62728", zorder=6, s=45, label=f"Operating Pt ({vin_dc:+.1f} V, {il_actual_ma:+.2f} mA)")
        style_cro_axes(
            ax_vi,
            title=f"Transfer Characteristic: $I_L$ vs. $V_{{in}}$ (Slope = $1/R = {gm_ma_per_v:.3f}$ mA/V)",
            xlabel="Input Voltage $V_{in}$ (V)",
            ylabel="$I_L$ (mA)"
        )
        ax_vi.legend(loc="upper left", fontsize=8)

        # Bottom Plot: Load Regulation (IL vs. RL at fixed Vin)
        rl_sw_kohm = np.linspace(0.1, max(20.0, r_kohm * 1.5), 400)
        rl_sw = rl_sw_kohm * 1e3
        vo_rl_id = 2.0 * (vin_dc / R) * rl_sw
        il_rl_act = np.where(
            np.abs(vo_rl_id) > vsat,
            ((vin_dc + np.sign(vo_rl_id) * vsat) / (2.0 * rl_sw + R)) * 1e3,
            (vin_dc / R) * 1e3
        )
        ax_rl.plot(rl_sw_kohm, il_rl_act, color="#2ca02c", linewidth=2.3, label=f"Load Current $I_L$ at $V_{{in}} = {vin_dc:+.1f}$ V")
        ax_rl.axvline(r_kohm, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"$R_L = R = {r_kohm:.1f}$ kOhm limit")
        ax_rl.scatter([rl_kohm], [il_actual_ma], color="#d62728", zorder=6, s=45)
        y_rl_max = max(abs((vin_dc / R) * 1e3) * 1.4, 0.2)
        ax_rl.set_ylim(-y_rl_max, y_rl_max)
        style_cro_axes(
            ax_rl,
            title=f"Load Regulation: $I_L$ vs. Grounded Load $R_L$ (Constant $I_L$ for $R_L \\le R$)",
            xlabel="Grounded Load Resistance $R_L$ (kOhm)",
            ylabel="$I_L$ (mA)"
        )
        ax_rl.legend(loc="lower left", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_tr)
        plt.close(fig_tr)

    tab_cro, tab_obs, tab_theory = st.tabs([
        "AC Voltage-to-Current Waveforms (CRO)",
        "Observation Tables (IL vs. Vin & IL vs. RL)",
        "Theory & Derivation (Gayakwad)"
    ])

    with tab_cro:
        T = 1.0 / freq_hz
        t = np.linspace(0.0, 3.0 * T, 2000)
        w = 2.0 * np.pi * freq_hz
        vin_t = vin_ac_peak * np.sin(w * t)
        vo_t_id = 2.0 * (vin_t / R) * RL
        il_t_ma = np.where(
            np.abs(vo_t_id) > vsat,
            ((vin_t + np.sign(vo_t_id) * vsat) / (2.0 * RL + R)) * 1e3,
            (vin_t / R) * 1e3
        )

        fig_cro, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(8.8, 4.6), sharex=True)
        t_ms = t * 1000.0

        ax_v.plot(t_ms, vin_t, color="#1f77b4", linewidth=2.1, label=f"Input Voltage $v_{{in}}(t)$ ($V_p = {vin_ac_peak:.1f}$ V)")
        ax_v.set_ylim(-max(vin_ac_peak * 1.3, 1.0), max(vin_ac_peak * 1.3, 1.0))
        style_cro_axes(
            ax_v,
            title=f"CH1: AC Input Voltage $v_{{in}}(t)$ ($f = {freq_hz}$ Hz)",
            xlabel="",
            ylabel="$v_{in}$ (V)"
        )
        ax_v.legend(loc="upper right", fontsize=8.5)

        ax_i.plot(t_ms, il_t_ma, color="#d62728", linewidth=2.2, label=f"Load Current $i_L(t) = v_{{in}}(t)/R$ (In-phase, $R_L = {rl_kohm:.1f}$ kOhm)")
        y_i_lim = max(np.max(np.abs(il_t_ma)) * 1.35, 0.2)
        ax_i.set_ylim(-y_i_lim, y_i_lim)
        style_cro_axes(
            ax_i,
            title=f"CH2: Grounded-Load Current $i_L(t)$ ($I_{{L(pp)}} = {np.ptp(il_t_ma):.3f}$ mA)",
            xlabel="Time (ms)",
            ylabel="$i_L$ (mA)"
        )
        ax_i.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_cro)
        plt.close(fig_cro)

    with tab_obs:
        col_t1, col_t2 = st.columns(2, gap="medium")
        with col_t1:
            st.markdown(f"**Table A: $I_L$ vs. $V_{{in}}$ ($R = {r_kohm:.1f}$ kOhm, $R_L = {rl_kohm:.1f}$ kOhm)**")
            v_vals = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
            rows_a = []
            for vv in v_vals:
                il_id = (vv / R) * 1e3
                vo_id = 2.0 * (vv / R) * RL
                il_ac = ((vv + np.sign(vo_id) * vsat) / (2.0 * RL + R)) * 1e3 if abs(vo_id) > vsat else il_id
                rows_a.append({
                    "Vin (V)": f"{vv:+.1f}",
                    "Ideal IL (mA)": f"{il_id:+.3f}",
                    "Actual IL (mA)": f"{il_ac:+.3f}",
                    "Status": "Saturated" if abs(vo_id) > vsat else "Linear"
                })
            st.table(rows_a)

        with col_t2:
            st.markdown(f"**Table B: Load Independence $I_L$ vs. $R_L$ ($V_{{in}} = {vin_dc:+.1f}$ V, $R = {r_kohm:.1f}$ kOhm)**")
            rl_vals = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0]
            rows_b = []
            for rlv in rl_vals:
                rl_ohm = rlv * 1e3
                il_id = (vin_dc / R) * 1e3
                vo_id = 2.0 * (vin_dc / R) * rl_ohm
                il_ac = ((vin_dc + np.sign(vo_id) * vsat) / (2.0 * rl_ohm + R)) * 1e3 if abs(vo_id) > vsat else il_id
                rows_b.append({
                    "Load RL (kOhm)": f"{rlv:.1f}",
                    "Ideal IL (mA)": f"{il_id:+.3f}",
                    "Actual IL (mA)": f"{il_ac:+.3f}",
                    "Vo (V)": f"{np.clip(vo_id, -vsat, vsat):+.2f}"
                })
            st.table(rows_b)

    with tab_theory:
        st.markdown(
            f"""
            ### Theory & Derivation (R. A. Gayakwad)

            1. **Kirchhoff's Current Law (KCL) at Node $V_1$:**
               Assuming ideal op-amp input currents $I_B \\cong 0$, summing the currents entering non-inverting node $V_1$ gives:
               $$I_1 + I_2 = I_L \\implies \\frac{{V_{{in}} - V_1}}{{R}} + \\frac{{V_o - V_1}}{{R}} = I_L$$
               $$V_{{in}} + V_o - 2 V_1 = I_L R \\implies V_1 = \\frac{{V_{{in}} + V_o - I_L R}}{{2}} \\quad \\text{{(Eq. 6-21a)}}$$

            2. **Load Current Derivation (*Gayakwad Eq. 6-21b*):**
               Because the op-amp is connected in the non-inverting mode with equal feedback resistors $R$, its closed-loop voltage gain is $1 + \\frac{{R}}{{R}} = 2$, so $V_o = 2 V_1$. Substituting $V_o = 2 V_1$ into Eq. (6-21a):
               $$2 V_1 = V_{{in}} + V_o - I_L R \\implies V_{{in}} = I_L R \\implies I_L = \\frac{{V_{{in}}}}{{R}} \\quad \\text{{(Eq. 6-21b)}}$$

            3. **Design Rules & Applications:**
               * **Equal Bridge Resistors:** All four bridge resistors $R$ must be matched in value ($1\\%$ tolerance) so that $I_L$ is completely independent of the grounded load resistance $R_L$[cite: 13].
               * **Load Size Limit ($R_L \\le R$):** Since $V_o = 2 V_1 = 2 I_L R_L$, the circuit performs satisfactorily provided the load size satisfies **$R_L \\le R$** (preventing op-amp output saturation at $\\pm V_{{sat}}$)[cite: 13].
               * **Applications:** Driving grounded **LEDs**, **Zener diode testers**, **analog meter movements**, and **4–20 mA industrial current transmitters**[cite: 13].
            """
        )
