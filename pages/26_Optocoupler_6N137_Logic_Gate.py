import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from utils import render_sidebar_header, style_cro_axes

st.set_page_config(page_title="Exp 26: Optocoupler 6N137 as Logic Gate", layout="wide")
render_sidebar_header()

st.title("Experiment 26: High-Speed Optocoupler (6N137) as a Logic Gate")
st.markdown(
    "**Aim:** To understand the pin configuration and working of a high-speed optocoupler (**6N137**), "
    "configure it as an **Inverting Logic Gate** and a **Non-Inverting Buffer**, and verify the truth table "
    "using an output LED indicator."
)

# -------------------------------------------------------------------------
# Helper: Draw White-Background 6N137 Optocoupler Circuit Schematic
# -------------------------------------------------------------------------
def draw_6n137_schematic(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm, led_on):
    with schemdraw.Drawing(show=False, bgcolor="white") as d:
        d.config(unit=2.45, fontsize=8.2, color="#0f172a", lw=1.6)

        # Central 8-pin DIP representation for 6N137
        # Left pins: Pin 2 (Anode), Pin 3 (Cathode)
        # Right pins: Pin 8 (VCC), Pin 7 (VE Enable), Pin 6 (VO Output), Pin 5 (GND)
        ic = d.add(
            elm.Ic(
                pins=[
                    elm.IcPin(name="2 (A)", side="left", pin="2"),
                    elm.IcPin(name="3 (K)", side="left", pin="3"),
                    elm.IcPin(name="5 (GND)", side="right", pin="5"),
                    elm.IcPin(name="6 (VO)", side="right", pin="6"),
                    elm.IcPin(name="7 (VE)", side="right", pin="7"),
                    elm.IcPin(name="8 (VCC)", side="right", pin="8"),
                ],
                w=3.2,
                h=4.2,
                pinspacing=0.95,
                label="6N137\nHigh-Speed\nOptocoupler",
                fontsize=8.2,
            ).at((0, 0))
        )

        p2 = ic.inL1  # Pin 2 (Anode)
        p3 = ic.inL2  # Pin 3 (Cathode)
        p5 = ic.inR1  # Pin 5 (GND)
        p6 = ic.inR2  # Pin 6 (Output VO)
        p7 = ic.inR3  # Pin 7 (Enable VE)
        p8 = ic.inR4  # Pin 8 (VCC +5V)

        # ----------------- INPUT SIDE (ISOLATED) -----------------
        if config_mode.startswith("Inverting"):
            # Standard Inverting Gate: VIN -> RIN -> Pin 2 (Anode); Pin 3 (Cathode) -> GND
            vin_label = "VIN = 5V (HIGH)" if vin_high else "VIN = 0V (LOW)"
            d.add(
                elm.Resistor()
                .left()
                .at(p2)
                .length(1.85)
                .label(f"$R_{{IN}}$\n{rin_ohm:.0f} $\\Omega$", loc="top", fontsize=8.0)
            )
            in_x = d.here[0]
            d.add(elm.Dot().at((in_x, p2[1])).label(vin_label, loc="left", fontsize=8.0))

            # Pin 3 (Cathode) to Input Ground
            d.add(elm.Line().left().at(p3).length(0.65))
            gnd_in_x = d.here[0]
            d.add(elm.Line().down().length(0.65))
            d.add(elm.Ground().label("GND1", loc="bottom", fontsize=7.5))
        else:
            # Non-Inverting Buffer (Cathode-Driven / Current-Sinking):
            # +5V -> RIN -> Pin 2 (Anode); Pin 3 (Cathode) -> VIN (LOW=ON, HIGH=OFF)
            d.add(
                elm.Resistor()
                .left()
                .at(p2)
                .length(1.65)
                .label(f"$R_{{IN}}$\n{rin_ohm:.0f} $\\Omega$", loc="top", fontsize=8.0)
            )
            v_in_top = d.here[0]
            d.add(elm.Line().up().length(0.45).label("+5V (Input)", loc="top", fontsize=7.8))

            vin_label = "VIN = 5V (HIGH)" if vin_high else "VIN = 0V (LOW)"
            d.add(elm.Line().left().at(p3).length(1.65))
            d.add(elm.Dot().label(vin_label, loc="left", fontsize=8.0))

        # ----------------- OUTPUT SIDE (ISOLATED) -----------------
        # Pin 8: +5V VCC & 0.1 uF bypass capacitor to Pin 5 (GND)
        vcc_x = p8[0] + 0.95
        vcc_rail_y = p8[1] + 0.85
        d.add(elm.Line().at(p8).to((vcc_x, p8[1])))
        d.add(elm.Dot().at((vcc_x, p8[1])))
        d.add(elm.Line().at((vcc_x, p8[1])).to((vcc_x, vcc_rail_y)).label("$V_{CC} = +5$ V", loc="top", fontsize=8.2))

        # 0.1 uF bypass capacitor from VCC rail to GND
        cap_x = vcc_x + 2.35
        gnd_out_y = p5[1] - 0.85
        d.add(elm.Line().at((vcc_x, vcc_rail_y)).to((cap_x, vcc_rail_y)))
        d.add(elm.Capacitor().at((cap_x, vcc_rail_y)).to((cap_x, gnd_out_y)).label("0.1 $\\mu$F", loc="right", fontsize=7.8))

        # Pin 5: Output Ground (GND2)
        d.add(elm.Line().at(p5).to((cap_x, p5[1])))
        d.add(elm.Dot().at((cap_x, p5[1])))
        d.add(elm.Line().at((cap_x, p5[1])).to((cap_x, gnd_out_y)))
        d.add(elm.Ground().at((cap_x, gnd_out_y)).label("GND2", loc="bottom", fontsize=7.5))

        # Pin 7: Enable (VE) -> H (+5V), L (GND), or NC (Open)
        ve_x = p7[0] + 0.55
        d.add(elm.Line().at(p7).to((ve_x, p7[1])))
        if enable_state == "H (+5V)":
            d.add(elm.Line().at((ve_x, p7[1])).to((ve_x, vcc_rail_y)))
            d.add(elm.Dot().at((ve_x, vcc_rail_y)))
        elif enable_state == "L (GND)":
            d.add(elm.Line().at((ve_x, p7[1])).to((ve_x, gnd_out_y)))
            d.add(elm.Dot().at((ve_x, gnd_out_y)))
        else:
            d.add(elm.Dot(open=True).at((ve_x, p7[1])).label("NC (Open)", loc="right", fontsize=7.6))

        # Pin 6: Open-Collector Output (VO) with Pull-Up RPULL to +5V and LED indicator to GND
        out_node_x = p6[0] + 1.45
        d.add(elm.Line().at(p6).to((out_node_x, p6[1])))
        d.add(elm.Dot().at((out_node_x, p6[1])))

        # Pull-up resistor RPULL from Pin 6 to +5V VCC rail
        d.add(
            elm.Resistor()
            .at((out_node_x, vcc_rail_y))
            .to((out_node_x, p6[1]))
            .label(f"$R_{{PULL}}$\n{rpull_ohm:.0f} $\\Omega$", loc="right", fontsize=7.8)
        )
        d.add(elm.Dot().at((out_node_x, vcc_rail_y)))

        # Output Indicator Branch: Pin 6 -> RLED (220 Ohm) -> Indicator LED -> GND
        led_x = out_node_x + 0.55
        d.add(elm.Line().at((out_node_x, p6[1])).to((led_x, p6[1])))
        mid_led_y = (p6[1] + gnd_out_y) / 2.0
        d.add(
            elm.Resistor()
            .at((led_x, p6[1]))
            .to((led_x, mid_led_y))
            .label(f"$R_{{LED}}$\n{rled_ohm:.0f} $\\Omega$", loc="right", fontsize=7.6)
        )
        led_state_lbl = "LED ON (HIGH)" if led_on else "LED OFF (LOW)"
        d.add(
            elm.LED()
            .at((led_x, mid_led_y))
            .to((led_x, gnd_out_y))
            .label(led_state_lbl, loc="right", fontsize=7.8)
        )
        d.add(elm.Dot().at((led_x, gnd_out_y)))

        buf = io.BytesIO()
        d.save(buf, dpi=220, transparent=False)
        buf.seek(0)
        plt.close("all")
        return buf

# -------------------------------------------------------------------------
# Physics & Logic Model for 6N137 High-Speed Optocoupler
# -------------------------------------------------------------------------
def evaluate_6n137(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm):
    vcc = 5.0
    vf_ir = 1.50        # Input IR LED forward drop (~1.5 V)
    if_thresh_ma = 5.0  # 6N137 input threshold current (~5 mA)
    vol_sat = 0.25      # Open-collector output LOW saturation voltage (~0.25 V)
    vf_ind = 2.00       # Output red indicator LED forward drop (~2.0 V)

    # Determine whether internal IR LED is ON
    if config_mode.startswith("Inverting"):
        # Active-HIGH input drives Anode (Pin 2) with Cathode (Pin 3) grounded
        ir_driven = vin_high
    else:
        # Non-inverting buffer: Anode pulled to +5V via RIN, Cathode driven by VIN (LOW turns IR LED ON)
        ir_driven = not vin_high

    if_ma = ((vcc - vf_ir) / rin_ohm) * 1e3 if ir_driven else 0.0
    ir_led_on = ir_driven and (if_ma >= if_thresh_ma)

    # Enable logic: Pin 7 (VE) is active-low in truth table:
    # Enable = H or NC -> Output enabled (if IR LED ON -> OUT = LOW; if IR LED OFF -> OUT = HIGH)
    # Enable = L -> Output forced disabled / high-impedance (OUT pulled HIGH via RPULL)
    if enable_state == "L (GND)":
        out_high = True
    else:
        out_high = not ir_led_on

    if out_high:
        # Open-collector transistor OFF: RPULL (1k) + RLED (220) + Indicator LED form divider to +5V
        i_led_ma = max((vcc - vf_ind) / (rpull_ohm + rled_ohm), 0.0) * 1e3
        vout_v = vf_ind + (i_led_ma * 1e-3) * rled_ohm
        ind_led_on = i_led_ma > 1.0
    else:
        # Open-collector transistor ON: Pin 6 pulled to VOL (~0.25 V)
        vout_v = vol_sat
        i_led_ma = 0.0
        ind_led_on = False

    return {
        "ir_led_on": ir_led_on,
        "if_ma": if_ma,
        "out_high": out_high,
        "vout_v": vout_v,
        "i_led_ma": i_led_ma,
        "ind_led_on": ind_led_on,
    }

# -------------------------------------------------------------------------
# Main Layout: Left Panel (Controls) | Right Panel (Circuit + Truth Table)
# -------------------------------------------------------------------------
ctrl_col, disp_col = st.columns([1.0, 2.3], gap="large")

with ctrl_col:
    st.subheader("1. Gate Configuration")
    config_mode = st.radio(
        "Select 6N137 Logic Configuration",
        [
            "Inverting Logic Gate (Standard Anode Drive)",
            "Non-Inverting Buffer (Cathode Drive)",
        ],
        index=0,
    )

    st.subheader("2. Digital Input & Enable Switches")
    vin_choice = st.radio("Digital Input $V_{\\text{IN}}$ (Pin 2 / Pin 3)", ["LOW (0 V)", "HIGH (+5 V)"], horizontal=True, index=1)
    vin_high = vin_choice.startswith("HIGH")

    enable_state = st.selectbox(
        "Enable Input $V_E$ (Pin 7)",
        ["H (+5V)", "L (GND)", "NC (Open)"],
        index=0,
        help="Pin 7 (Enable): When H (+5V) or NC (Open), output is enabled. When L (GND), output is forced HIGH."
    )

    st.subheader("3. Resistor Values")
    rin_ohm = st.slider(
        "Input Current-Limiting Resistor $R_{\\text{IN}}$ (Ohm)",
        min_value=220,
        max_value=1000,
        value=470,
        step=10,
        help="Lab manual design: RIN = (5V - 1.5V) / 10 mA = 350 Ohm; 470 Ohm is standard (IF = 7.45 mA)."
    )
    rpull_ohm = st.slider("Output Pull-Up Resistor $R_{\\text{PULL}}$ (Ohm)", 470, 4700, 1000, 100)
    rled_ohm = st.slider("Indicator LED Resistor $R_{\\text{LED}}$ (Ohm)", 150, 470, 220, 10)

# -------------------------------------------------------------------------
# Evaluate Logic State
# -------------------------------------------------------------------------
res = evaluate_6n137(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm)

# -------------------------------------------------------------------------
# Right Display Panel
# -------------------------------------------------------------------------
with disp_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internal IR LED", "ON" if res["ir_led_on"] else "OFF")
    m2.metric("Input Forward Current $I_F$", f"{res['if_ma']:.2f} mA")
    m3.metric("Output Logic (Pin 6)", "HIGH (H)" if res["out_high"] else "LOW (L)")
    m4.metric("Output Voltage $V_O$", f"{res['vout_v']:.2f} V")

    if res["ind_led_on"]:
        st.success(
            f"**Output HIGH ({res['vout_v']:.2f} V) -- Indicator LED ON ($I_{{\\text{{LED}}}} = {res['i_led_ma']:.2f}$ mA):** "
            f"Internal IR LED is **{'ON' if res['ir_led_on'] else 'OFF'}** ($I_F = {res['if_ma']:.2f}$ mA), Enable (Pin 7) = **{enable_state}**."
        )
    else:
        st.info(
            f"**Output LOW ({res['vout_v']:.2f} V) -- Indicator LED OFF ($I_{{\\text{{LED}}}} = 0.00$ mA):** "
            f"Internal IR LED is **ON** ($I_F = {res['if_ma']:.2f}$ mA) and Enable (Pin 7) = **{enable_state}**, pulling Pin 6 LOW."
        )

    schem_col, tt_col = st.columns([1.25, 1.25], gap="medium")

    with schem_col:
        st.markdown(f"**6N137 Schematic ({config_mode.split(' (')[0]})**")
        sch_buf = draw_6n137_schematic(
            config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm, res["ind_led_on"]
        )
        st.image(sch_buf, use_container_width=True)
        st.caption(
            "Galvanic isolation between input side (Pins 2 & 3, GND1) and output side (Pins 5–8, GND2) with $0.1\\;\\mu\\text{F}$ bypass capacitor."
        )

    with tt_col:
        st.markdown("**Truth Table Verification (6N137)**")
        truth_rows = [
            {"LED": "ON",  "Enable (Pin 7)": "H",  "Out (Pin 6)": "L", "Indicator LED": "OFF"},
            {"LED": "OFF", "Enable (Pin 7)": "H",  "Out (Pin 6)": "H", "Indicator LED": "ON"},
            {"LED": "ON",  "Enable (Pin 7)": "L",  "Out (Pin 6)": "H", "Indicator LED": "ON"},
            {"LED": "OFF", "Enable (Pin 7)": "L",  "Out (Pin 6)": "H", "Indicator LED": "ON"},
            {"LED": "ON",  "Enable (Pin 7)": "NC", "Out (Pin 6)": "L", "Indicator LED": "OFF"},
            {"LED": "OFF", "Enable (Pin 7)": "NC", "Out (Pin 6)": "H", "Indicator LED": "ON"},
        ]

        cur_led = "ON" if res["ir_led_on"] else "OFF"
        cur_en = enable_state.split(" ")[0]
        for row in truth_rows:
            if row["LED"] == cur_led and row["Enable (Pin 7)"] == cur_en:
                row["Active State"] = "<-- CURRENT"
            else:
                row["Active State"] = ""

        st.table(truth_rows)

        st.markdown("**Logic Gate Summary:**")
        st.markdown(
            """
            * **Inverting Gate (Pin 2 driven, Pin 3 grounded):**
              $V_{\\text{IN}} = \\text{HIGH} \\Rightarrow \\text{LED ON} \\Rightarrow V_O = \\text{LOW}$ (NOT Gate when Enable = H or NC).
            * **Non-Inverting Buffer (+5 V on Pin 2, Pin 3 driven):**
              $V_{\\text{IN}} = \\text{HIGH} \\Rightarrow \\text{LED OFF} \\Rightarrow V_O = \\text{HIGH}$ (Buffer Gate when Enable = H or NC).
            """
        )

    # ---------------------------------------------------------------------
    # Secondary Tabs: Input Current Design Curve + Lab Manual Theory
    # -------------------------------------------------------------------------
    tab_curve, tab_manual = st.tabs([
        "Input Resistor Design Curve (IF vs. RIN)",
        "Aim, Pin Configuration, Theory & Procedure"
    ])

    with tab_curve:
        rin_sweep = np.linspace(150, 1000, 300)
        if_sweep = ((5.0 - 1.5) / rin_sweep) * 1e3

        fig_if, ax_if = plt.subplots(figsize=(8.6, 3.6))
        ax_if.plot(rin_sweep, if_sweep, color="#1f77b4", linewidth=2.2, label="$I_F = (V_{IN} - V_F) / R_{IN}$ ($V_{IN}=5$ V, $V_F=1.5$ V)")
        ax_if.axhspan(5.0, 15.0, color="#2ca02c", alpha=0.14, label="Recommended 6N137 Forward Current Range (5–15 mA)")
        ax_if.scatter([rin_ohm], [(5.0 - 1.5) / rin_ohm * 1e3], color="#d62728", zorder=6, s=55, label=f"Selected $R_{{IN}} = {rin_ohm}\\;\\Omega$ ($I_F = {(5.0-1.5)/rin_ohm*1e3:.2f}$ mA)")
        style_cro_axes(ax_if, title="6N137 Input LED Forward Current IF vs. Limiting Resistor RIN", xlabel="Input Resistor RIN (Ohm)", ylabel="Forward Current IF (mA)")
        ax_if.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_if, clear_figure=True)
        plt.close(fig_if)

    with tab_manual:
        st.markdown(
            f"""
            ### Laboratory Write-Up & Pin Configuration

            * **Pin Configuration of 6N137 (8-Pin DIP):**
              * **Pins 1 & 4 (NC):** No Connection.
              * **Pin 2 (Anode) & Pin 3 (Cathode):** Input infrared LED ($I_F = 5\\text{{--}}15\\text{{ mA}}$, $V_F \\approx 1.5\\text{{ V}}$).
              * **Pin 5 (GND):** Output-side ground.
              * **Pin 6 (Output $V_O$):** Open-collector logic output (requires pull-up resistor $R_{{PULL}} = 1\\text{{ k}}\\Omega$ to $V_{{CC}}$).
              * **Pin 7 (Enable $V_E$):** Output enable input (when left open/`NC` or connected `HIGH`, output is enabled; when connected `LOW`, output is disabled / high-impedance).
              * **Pin 8 ($V_{{CC}}$):** Output-side supply ($+5\\text{{ V}}$ typical), decoupled to Pin 5 with a $0.1\\;\\mu\\text{{F}}$ ceramic bypass capacitor.

            * **Input Resistor Design Calculation:**
              For $V_{{IN}} = 5\\text{{ V}}$, $V_F \\approx 1.5\\text{{ V}}$, and desired $I_F = 10\\text{{ mA}}$:
              $$R_{{IN}} = \\frac{{5\\text{{ V}} - 1.5\\text{{ V}}}}{{0.01\\text{{ A}}}} = 350\\;\\Omega$$
              A standard **$470\\;\\Omega$** resistor provides a safe forward current of $I_F = \\frac{{3.5\\text{{ V}}}}{{470\\;\\Omega}} = 7.45\\text{{ mA}}$ (well within the $5\\text{{--}}15\\text{{ mA}}$ operating window).
            """
        )