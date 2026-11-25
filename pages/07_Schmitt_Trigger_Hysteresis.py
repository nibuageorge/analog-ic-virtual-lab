import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes, format_eng

st.set_page_config(page_title="Exp 7: Schmitt Trigger & Hysteresis", layout="wide")
render_sidebar_header()

st.title("Experiment 7: Schmitt Trigger & Hysteresis Curve using Op-Amp ($\mu$A741)")
st.markdown(
    "Design and construct an inverting comparator with positive feedback (**Schmitt Trigger**) using **$\mu$A741**, "
    "observe the input/output waveforms, and plot the **$v_o$ vs. $v_{in}$ hysteresis curve** "
    "(*Gayakwad, Section 8.3, Figure 8-4*)."
)

# -------------------------------------------------------------------------
# Helper: Draw Exact Gayakwad Figure 8-4(a) Schmitt Trigger Schematic
# -------------------------------------------------------------------------
def draw_schmitt_trigger_schematic(r1_ohm, r2_kohm, rl_kohm, rom_ohm, vcc):
    r1_label = f"{r1_ohm/1000:.2f} kΩ" if r1_ohm >= 1000 else f"{r1_ohm:.0f} Ω"
    rom_label = f"{rom_ohm/1000:.2f} kΩ" if rom_ohm >= 1000 else f"{rom_ohm:.0f} Ω"

    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=8.5)

        # --- Op-Amp (741/351): Inverting (-) at top, Non-Inverting (+) at bottom ---
        op = d.add(elm.Opamp(leads=True).at((0, 0)).label('741/351', loc='center', fontsize=8.5))
        d.add(elm.Line().up().at(op.vd).length(0.35).label(f'+{vcc:.0f}V', loc='top', fontsize=7.5))
        d.add(elm.Line().down().at(op.vs).length(0.35).label(f'-{vcc:.0f}V', loc='bottom', fontsize=7.5))

        # --- Inverting Input Branch (-): R_OM = R1 || R2 and AC Source v_in ---
        inv_x, inv_y = op.in1
        d.add(elm.Line().left().at((inv_x, inv_y)).length(0.65))
        d.add(elm.Resistor().left().length(1.65).label(f'$R_{{OM}} = R_1 \\parallel R_2$\n{rom_label}', loc='top', fontsize=8))
        d.add(elm.Line().down().length(0.35))
        src = d.add(elm.SourceSin().down().length(1.15).label('$v_{in}$', loc='left', fontsize=9))
        d.add(elm.Ground())

        # --- Output Node & Load Resistor R_L ---
        d.add(elm.Line().right().at(op.out).length(0.65))
        fb_tap = d.add(elm.Dot())
        fb_x, fb_y = fb_tap.center

        d.add(elm.Line().right().at((fb_x, fb_y)).length(0.85))
        rl_tap = d.add(elm.Dot())
        rl_x, rl_y = rl_tap.center

        d.add(elm.Line().right().at((rl_x, rl_y)).length(0.65).label('$v_o$', loc='right', fontsize=9.5))

        # Load Resistor RL to Ground
        d.add(elm.Resistor().down().at((rl_x, rl_y)).length(1.55).label(f'$R_L$\n{rl_kohm:.1f} kΩ', loc='right', fontsize=8.5))
        d.add(elm.Ground())

        # --- Positive Feedback Voltage Divider (R2 & R1) to Non-Inverting (+) Pin ---
        noninv_x, noninv_y = op.in2
        vft_y = noninv_y - 1.25
        d.add(elm.Line().left().at((noninv_x, noninv_y)).length(0.65))
        d.add(elm.Line().down().toy(vft_y))
        vft_node = d.add(elm.Dot().label('$V_{ft}$', loc='left', fontsize=9))
        vx, vy = vft_node.center

        # Resistor R1 from V_ft node down to Ground
        d.add(elm.Resistor().down().at((vx, vy)).length(1.35).label(f'$R_1$\n{r1_label}', loc='right', fontsize=8.5))
        d.add(elm.Ground())

        # Resistor R2 from V_ft node right to Output feedback tap
        d.add(elm.Resistor().right().at((vx, vy)).tox(fb_x).label(f'$R_2$\n{r2_kohm:.1f} kΩ', loc='top', fontsize=8.5))
        d.add(elm.Line().up().toy(fb_y))

        fig = d.draw(show=False).fig
        return fig

# -------------------------------------------------------------------------
# Helper: Simulate Schmitt Trigger Waveforms & Hysteresis Loop
# -------------------------------------------------------------------------
def simulate_schmitt_trigger(R1, R2, v_peak, freq, vcc, num_cycles=2, num_pts=4000):
    vsat = max(vcc - 1.0, 0.5)  # Gayakwad Fig. 8-4(b): ±Vsat ≈ ±14 V for ±15 V supply

    # Gayakwad Section 8.3 Threshold & Hysteresis Equations
    beta = R1 / (R1 + R2)
    v_ut = beta * vsat
    v_lt = -beta * vsat
    v_hy = v_ut - v_lt

    T = 1.0 / freq
    t = np.linspace(0.0, num_cycles * T, num_pts)
    vin = v_peak * np.sin(2.0 * np.pi * freq * t)

    # Sequential state simulation of inverting Schmitt trigger
    vo = np.empty_like(vin)
    # Start in +Vsat state if initial vin < V_ut, else -Vsat
    state = vsat if vin[0] < v_ut else -vsat
    for i, v_val in enumerate(vin):
        if state > 0 and v_val >= v_ut:
            state = -vsat
        elif state < 0 and v_val <= v_lt:
            state = vsat
        vo[i] = state

    triggered = v_peak > v_ut
    return t, vin, vo, v_ut, v_lt, v_hy, vsat, triggered

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + CRO + XY)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Feedback Divider ($R_1, R_2$)")
    r1_ohm = st.slider(
        "Ground Resistor $R_1$ (Ω)",
        min_value=50,
        max_value=5000,
        value=100,
        step=50,
        help="Gayakwad Example 8-4 default: R1 = 100 Ω, R2 = 56 kΩ -> Vut = +25 mV, Vlt = -25 mV."
    )
    r2_kohm = st.slider(
        "Feedback Resistor $R_2$ (kΩ)",
        min_value=1.0,
        max_value=100.0,
        value=56.0,
        step=1.0,
        help="Controls positive feedback fraction R1 / (R1 + R2) and hysteresis width."
    )
    rl_kohm = st.slider(
        "Load Resistor $R_L$ (kΩ)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Textbook load resistor RL = 10 kΩ."
    )

    st.subheader("2. Sinusoidal Input Signal ($v_{in}$)")
    v_peak = st.slider(
        "Input Peak Amplitude $V_p$ (V)",
        min_value=0.05,
        max_value=5.00,
        value=0.50,
        step=0.05,
        help="Must exceed Vut to trigger output switching (Gayakwad Fig. 8-4b default: Vp = 0.5 V)."
    )
    freq_hz = st.slider(
        "Input Frequency $f$ (Hz)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

    st.subheader("3. Supply & Display Settings")
    vcc = st.slider("Supply Voltage $\pm V_{CC} / \pm V_{EE}$ (V)", min_value=5.0, max_value=18.0, value=15.0, step=1.0)
    num_cycles = st.slider("Cycles Displayed on CRO", min_value=1, max_value=5, value=2, step=1)

# -------------------------------------------------------------------------
# Circuit Calculations & Simulation
# -------------------------------------------------------------------------
R1 = float(r1_ohm)
R2 = r2_kohm * 1e3
rom_ohm = (R1 * R2) / (R1 + R2)

t_disp, vin_disp, vo_disp, v_ut, v_lt, v_hy, vsat, triggered = simulate_schmitt_trigger(
    R1, R2, v_peak, freq_hz, vcc, num_cycles=num_cycles
)

# -------------------------------------------------------------------------
# Right Display Panel: Metrics, Schematic, Waveforms, and Hysteresis Loop
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Upper Threshold ($V_{ut}$)", format_eng(v_ut, "V"))
    m2.metric("Lower Threshold ($V_{lt}$)", format_eng(v_lt, "V"))
    m3.metric("Hysteresis Width ($V_{hy}$)", format_eng(v_hy, "V"))
    m4.metric("Output Swing ($\\pm V_{sat}$)", f"±{vsat:.1f} V")

    if not triggered:
        st.error(
            f"**Input Below Trigger Threshold ($V_p = {v_peak:.2f}\\text{{ V}} \\le V_{{ut}} = {v_ut:.3f}\\text{{ V}}$):** "
            f"The sinusoidal input never crosses $V_{{ut}}$ or $V_{{lt}}$, so the output remains latched at $+V_{{sat}}$. "
            f"Increase input peak amplitude $V_p$ above **{v_ut:.3f} V** (or reduce $R_1$)."
        )
    else:
        st.success(
            f"**Schmitt Trigger Switching Active ($V_p = {v_peak:.2f}\\text{{ V}} > V_{{ut}} = {format_eng(v_ut, 'V')}$):** "
            f"Output switches from $+{vsat:.1f}\\text{{ V}}$ to $-{vsat:.1f}\\text{{ V}}$ at $v_{{in}} = +{format_eng(v_ut, 'V')}$, "
            f"and switches back to $+{vsat:.1f}\\text{{ V}}$ at $v_{{in}} = {format_eng(v_lt, 'V')}$ (Hysteresis $V_{{hy}} = {format_eng(v_hy, 'V')}$)."
        )

    # Row 1: Circuit Diagram (Left) + Time-Domain Waveforms Fig. 8-4(b) (Right)
    schem_col, wave_col = st.columns([1.05, 1.45], gap="medium")

    with schem_col:
        st.markdown("**Circuit Diagram (*Gayakwad Fig. 8-4a*)**")
        fig_sch = draw_schmitt_trigger_schematic(r1_ohm, r2_kohm, rl_kohm, rom_ohm, vcc)
        st.pyplot(fig_sch)
        plt.close(fig_sch)
        st.caption(
            f"**Feedback Ratio:** $\\frac{{R_1}}{{R_1 + R_2}} = {R1/(R1+R2):.5f}$; "
            f"**Offset Resistor:** $R_{{OM}} = R_1 \\parallel R_2 = {format_eng(rom_ohm, 'Ω')}$."
        )

    with wave_col:
        st.markdown("**Input & Output Waveforms (*Gayakwad Fig. 8-4b*)**")
        fig_wave, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.5, 5.1), sharex=True)
        t_ms = t_disp * 1000.0

        # Top: Sinusoidal Input vin(t) with Vut and Vlt threshold lines
        ax_in.plot(t_ms, vin_disp, color="#1f77b4", linewidth=2.1, label="Input $v_{in}(t)$")
        ax_in.axhline(v_ut, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"$V_{{ut}} = +{format_eng(v_ut, 'V')}$")
        ax_in.axhline(v_lt, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"$V_{{lt}} = {format_eng(v_lt, 'V')}$")
        y_in_lim = max(v_peak, abs(v_ut)) * 1.35
        ax_in.set_ylim(-y_in_lim, y_in_lim)
        style_cro_axes(
            ax_in,
            title=f"CH1: Sinusoidal Input $v_{{in}}(t)$ ($V_p = {v_peak:.2f}$ V, $f = {freq_hz}$ Hz)",
            xlabel="",
            ylabel="$v_{in}$ (V)"
        )
        ax_in.legend(loc="upper right", fontsize=8, ncol=3)

        # Bottom: Square-Wave Output vo(t)
        ax_out.plot(t_ms, vo_disp, color="#d62728", linewidth=2.2, label="Output $v_o(t)$")
        ax_out.set_ylim(-vsat * 1.25, vsat * 1.25)
        style_cro_axes(
            ax_out,
            title=f"CH2: Schmitt Trigger Square-Wave Output $v_o(t)$ ($\\pm V_{{sat}} = \\pm {vsat:.1f}$ V)",
            xlabel="Time (ms)",
            ylabel="$v_o$ (V)"
        )
        ax_out.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

    # Row 2: Hysteresis Curve Plot (Fig. 8-4c) + Theory & Design Equations
    hyst_col, theory_col = st.columns([1.15, 1.35], gap="medium")

    with hyst_col:
        st.markdown("**Hysteresis Curve: $v_o$ vs. $v_{in}$ (*Gayakwad Fig. 8-4c*)**")
        fig_hyst, ax_h = plt.subplots(figsize=(6.2, 4.4))

        x_span = max(v_peak, abs(v_ut) * 1.6) * 1.15
        # Draw upper branch (+Vsat from -x_span to +Vut) and lower branch (-Vsat from +x_span to -Vlt)
        ax_h.plot([-x_span, v_ut], [vsat, vsat], color="#1f77b4", linewidth=2.4)
        ax_h.plot([v_ut, v_ut], [vsat, -vsat], color="#d62728", linewidth=2.2)
        ax_h.plot([v_ut, x_span], [-vsat, -vsat], color="#1f77b4", linewidth=2.4)
        ax_h.plot([x_span, v_lt], [-vsat, -vsat], color="#1f77b4", linewidth=2.4)
        ax_h.plot([v_lt, v_lt], [-vsat, vsat], color="#2ca02c", linewidth=2.2)
        ax_h.plot([v_lt, -x_span], [vsat, vsat], color="#1f77b4", linewidth=2.4)

        # Directional arrows matching Gayakwad Fig. 8-4(c)
        ax_h.annotate("", xy=(v_ut, 0.0), xytext=(v_ut, vsat * 0.45),
                      arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.2))
        ax_h.annotate("", xy=(v_lt, 0.0), xytext=(v_lt, -vsat * 0.45),
                      arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.2))
        ax_h.annotate("", xy=(0.0, vsat), xytext=(-x_span * 0.45, vsat),
                      arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.0))
        ax_h.annotate("", xy=(0.0, -vsat), xytext=(x_span * 0.45, -vsat),
                      arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.0))

        # Threshold & Hysteresis Annotations
        ax_h.scatter([v_lt, v_ut], [0, 0], color="black", zorder=5, s=30)
        ax_h.text(v_ut + 0.04 * x_span, vsat * 0.08, f"$V_{{ut}} = +{format_eng(v_ut, 'V')}$", fontsize=8.5, fontweight="bold", color="#d62728")
        ax_h.text(v_lt - 0.04 * x_span, -vsat * 0.16, f"$V_{{lt}} = {format_eng(v_lt, 'V')}$", fontsize=8.5, fontweight="bold", ha="right", color="#2ca02c")

        # Double-headed arrow indicating Hysteresis Voltage (Vut - Vlt)
        y_bar = -vsat * 0.48
        ax_h.annotate("", xy=(v_ut, y_bar), xytext=(v_lt, y_bar),
                      arrowprops=dict(arrowstyle="<->", color="#6a3d9a", lw=1.6))
        ax_h.text(0.0, y_bar - vsat * 0.16, f"Hysteresis Voltage\n$(V_{{ut}} - V_{{lt}}) = {format_eng(v_hy, 'V')}$",
                  ha="center", va="top", fontsize=8.5, color="#6a3d9a", fontweight="bold")

        ax_h.set_xlim(-x_span, x_span)
        ax_h.set_ylim(-vsat * 1.3, vsat * 1.3)
        style_cro_axes(
            ax_h,
            title="Transfer Characteristic ($v_o$ vs. $v_{in}$ in CRO X-Y Mode)",
            xlabel="Input Voltage $v_{in}$ (V)",
            ylabel="Output Voltage $v_o$ (V)"
        )
        plt.tight_layout()
        st.pyplot(fig_hyst)
        plt.close(fig_hyst)

    with theory_col:
        st.markdown("**Theory & Design Equations (*R. A. Gayakwad, Section 8.3*)**")
        st.markdown(
            f"""
            1. **Principle of Regenerative (Positive) Feedback:**
               * The input signal $v_{{in}}$ is applied to the **inverting input (`-`)** through $R_{{OM}} = R_1 \\parallel R_2$, while the output $v_o$ is fed back to the **non-inverting input (`+`)** through the voltage divider $R_1$ and $R_2$[cite: 7].
               * The feedback voltage $V_{{ft}}$ at the `+` terminal depends on the output state ($+V_{{sat}}$ or $-V_{{sat}}$)[cite: 7]:

            2. **Upper Threshold Voltage ($V_{{ut}}$):**
               When $v_o = +V_{{sat}}$, the output switches to $-V_{{sat}}$ as soon as $v_{{in}}$ rises slightly above:
               $$V_{{ut}} = \\frac{{R_1}}{{R_1 + R_2}} (+V_{{sat}}) = \\frac{{{R1:.0f}}}{{{R1:.0f} + {R2:.0f}}} (+{vsat:.1f}\\text{{ V}}) = +{format_eng(v_ut, 'V')}$$

            3. **Lower Threshold Voltage ($V_{{lt}}$):**
               When $v_o = -V_{{sat}}$, the output switches back to $+V_{{sat}}$ as soon as $v_{{in}}$ falls slightly below:
               $$V_{{lt}} = \\frac{{R_1}}{{R_1 + R_2}} (-V_{{sat}}) = \\frac{{{R1:.0f}}}{{{R1:.0f} + {R2:.0f}}} (-{vsat:.1f}\\text{{ V}}) = {format_eng(v_lt, 'V')}$$

            4. **Hysteresis Voltage ($V_{{hy}}$):**
               The dead-band width between the two switching thresholds eliminates false triggering due to input noise[cite: 7]:
               $$V_{{hy}} = V_{{ut}} - V_{{lt}} = \\frac{{2 R_1}}{{R_1 + R_2}} V_{{sat}} = {format_eng(v_hy, 'V')}$$
            """
        )