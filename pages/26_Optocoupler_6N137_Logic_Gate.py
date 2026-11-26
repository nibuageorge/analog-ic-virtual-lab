import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import schemdraw
import schemdraw.elements as elm

def draw_6n137_schematic(config_mode, vin_high, enable_state, rin_ohm, rpull_ohm, rled_ohm, led_on):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    d = schemdraw.Drawing(canvas=ax, show=False)
    d.config(unit=2.2, fontsize=8.2, color="#0f172a", lw=1.45)

    # ---------------------------------------------------------------------
    # 1. Central 6N137 IC Package (Matching uploaded schematic layout)
    # Left pins (top to bottom): 1 (N/C), 2 (+), 3 (-), 4 (N/C)
    # Right pins (top to bottom): 8 (VCC), 7 (VE), 6 (VO), 5 (GND)
    # ---------------------------------------------------------------------
    ic_left, ic_right = -1.65, 1.65
    ic_bot, ic_top = -2.35, 2.35
    y_p1, y_p2, y_p3, y_p4 = 1.45, 0.48, -0.48, -1.45

    # IC Outer Box
    ax.add_patch(
        patches.Rectangle(
            (ic_left, ic_bot),
            ic_right - ic_left,
            ic_top - ic_bot,
            linewidth=1.5,
            edgecolor="#0f172a",
            facecolor="white",
            zorder=2,
        )
    )

    # Pin Number Boxes & Labels
    left_pins = [(1, y_p1, "N/C", "+"), (2, y_p2, "", "+"), (3, y_p3, "", "-"), (4, y_p4, "N/C", "")]
    for pnum, py, ext_lbl, int_lbl in left_pins:
        ax.add_patch(patches.Rectangle((ic_left - 0.34, py - 0.18), 0.34, 0.36, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_left - 0.17, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        if ext_lbl:
            ax.text(ic_left - 0.44, py, ext_lbl, ha="right", va="center", fontsize=7.8, color="#0f172a")
        if int_lbl and pnum in (2, 3):
            ax.text(ic_left - 0.44, py, int_lbl, ha="right", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")

    ax.text(ic_left - 0.52, (y_p2 + y_p3) / 2.0, "$V_F$", ha="right", va="center", fontsize=8.2, color="#0f172a")

    right_pins = [(8, y_p1, "$V_{CC}$"), (7, y_p2, "$V_E$"), (6, y_p3, "$V_O$"), (5, y_p4, "GND")]
    for pnum, py, ext_lbl in right_pins:
        ax.add_patch(patches.Rectangle((ic_right, py - 0.18), 0.34, 0.36, lw=1.2, ec="#0f172a", fc="white", zorder=3))
        ax.text(ic_right + 0.17, py, str(pnum), ha="center", va="center", fontsize=7.5, color="#0f172a", zorder=4)
        ax.text(ic_right + 0.44, py + 0.08, ext_lbl, ha="left", va="bottom", fontsize=7.8, color="#0f172a")

    # ---------------------------------------------------------------------
    # 2. Internal 6N137 Schematic (IR LED + Isolation Barrier + Detector + Open Collector)
    # ---------------------------------------------------------------------
    led_in_x = -0.95
    d.add(elm.Line().at((ic_left, y_p2)).to((led_in_x, y_p2)))
    d.add(elm.Diode().at((led_in_x, y_p2)).to((led_in_x, y_p3)).fill(True))
    d.add(elm.Line().at((led_in_x, y_p3)).to((ic_left, y_p3)))

    # Optical arrows & dashed isolation barrier
    ax.annotate("", xy=(-0.38, 0.05), xytext=(-0.78, -0.05), arrowprops=dict(arrowstyle="->", lw=1.1, color="#0f172a"))
    ax.plot([-0.55, -0.55], [ic_bot + 0.25, ic_top - 0.25], linestyle="--", linewidth=1.1, color="#0f172a", zorder=3)

    # Detector side internal rails (VCC bus at y_p1, GND bus at y_p4)
    pd_x = -0.22
    amp_x = 0.25
    q_x = 1.05
    d.add(elm.Line().at((pd_x, y_p1)).to((ic_right, y_p1)))
    d.add(elm.Line().at((pd_x, y_p4)).to((ic_right, y_p4)))
    d.add(elm.Dot().at((pd_x, y_p1)))
    d.add(elm.Dot().at((amp_x, y_p1)))
    d.add(elm.Dot().at((amp_x, y_p4)))
    d.add(elm.Dot().at((q_x, y_p4)))

    # Internal Photodiode -> Buffer -> Gate -> Open-Collector NPN
    d.add(elm.Line().at((pd_x, y_p1)).to((pd_x, 0.35)))
    d.add(elm.Diode().at((pd_x, -0.45)).to((pd_x, 0.35)).fill(True))
    d.add(elm.Line().at((pd_x, -0.45)).to((pd_x, -1.05)))
    d.add(elm.Line().at((pd_x, -1.05)).to((amp_x, -1.05)))
    d.add(elm.Line().at((amp_x, y_p1)).to((amp_x, y_p4)))

    # Enable line from Pin 7 to Gate
    en_int_x = 0.58
    d.add(elm.Line().at((ic_right, y_p2)).to((en_int_x, y_p2)))
    d.add(elm.Line().at((en_int_x, y_p2)).to((en_int_x, -0.92)))
    d.add(elm.Line().at((en_int_x, -0.92)).to((0.72, -0.92)))

    # Small internal amplifier & gate symbols
    ax.add_patch(patches.Polygon([[0.12, -0.88], [0.12, -1.22], [0.42, -1.05]], closed=True, lw=1.1, ec="#0f172a", fc="white", zorder=4))
    ax.add_patch(patches.Rectangle((0.66, -1.16), 0.22, 0.28, lw=1.1, ec="#0f172a", fc="white", zorder=4))

    # Open-collector NPN output transistor between Pin 6 (VO) and Pin 5 (GND)
    d.add(elm.BjtNpn().at((q_x, -1.02)).anchor("base"))
    d.add(elm.Line().at((0.88, -1.02)).to((q_x, -1.02)))
    d.add(elm.Line().at((q_x + 0.36, -0.68)).to((q_x + 0.36, y_p3)))
    d.add(elm.Line().at((q_x + 0.36, y_p3)).to((ic_right, y_p3)))
    d.add(elm.Line().at((q_x + 0.36, -1.36)).to((q_x + 0.36, y_p4)))

    # ---------------------------------------------------------------------
    # 3. External Input Side (Left of Pins 2 & 3)
    # ---------------------------------------------------------------------
    pin2_ext = (ic_left - 0.34, y_p2)
    pin3_ext = (ic_left - 0.34, y_p3)
    bat_in_x = -6.15

    if config_mode.startswith("Inverting"):
        # 5V Battery -> 470 Ohm -> Switch S -> Pin 2 (+); Pin 3 (-) -> Battery (-)
        sw_r_x = -2.85
        sw_l_x = -3.85
        r_l_x = -5.45

        d.add(elm.Line().at(pin2_ext).to((sw_r_x, y_p2)))
        if vin_high:
            d.add(elm.Switch(contacts=True).at((sw_l_x, y_p2)).to((sw_r_x, y_p2)).label("S (Closed)", loc="top", fontsize=8.0))
        else:
            d.add(elm.Switch(contacts=True).at((sw_l_x, y_p2)).to((sw_r_x, y_p2)).label("S (Open)", loc="top", fontsize=8.0))
            # Overlay open lever explicitly
        d.add(
            elm.Resistor()
            .at((r_l_x, y_p2))
            .to((sw_l_x, y_p2))
            .label(f"{rin_ohm:.0f}$\\Omega$", loc="top", fontsize=8.5)
        )
        d.add(elm.Line().at((bat_in_x, y_p2)).to((r_l_x, y_p2)))
        d.add(elm.BatteryCell().at((bat_in_x, y_p3)).to((bat_in_x, y_p2)).label("5V", loc="left", fontsize=8.5))
        d.add(elm.Line().at((bat_in_x, y_p3)).to(pin3_ext))
    else:
        # Non-Inverting Buffer: 5V -> 470 Ohm -> Pin 2 (+); Pin 3 (-) -> Switch S -> Battery (-)
        r_l_x = -4.85
        sw_r_x = -2.85
        sw_l_x = -3.85
        d.add(elm.Line().at((bat_in_x, y_p2)).to((r_l_x, y_p2)))
        d.add(elm.Resistor().at((r_l_x, y_p2)).to(pin2_ext).label(f"{rin_ohm:.0f}$\\Omega$", loc="top", fontsize=8.5))
        d.add(elm.BatteryCell().at((bat_in_x, y_p3)).to((bat_in_x, y_p2)).label("5V", loc="left", fontsize=8.5))
        d.add(elm.Line().at((bat_in_x, y_p3)).to((sw_l_x, y_p3)))
        sw_lbl = "S (Closed=LOW)" if not vin_high else "S (Open=HIGH)"
        d.add(elm.Switch(contacts=True).at((sw_l_x, y_p3)).to((sw_r_x, y_p3)).label(sw_lbl, loc="bottom", fontsize=7.8))
        d.add(elm.Line().at((sw_r_x, y_p3)).to(pin3_ext))

    # ---------------------------------------------------------------------
    # 4. External Output Side (Right of Pins 8, 7, 6, 5)
    # ---------------------------------------------------------------------
    p8_ext = (ic_right + 0.34, y_p1)
    p7_ext = (ic_right + 0.34, y_p2)
    p6_ext = (ic_right + 0.34, y_p3)
    p5_ext = (ic_right + 0.34, y_p4)

    vcc_bus_x = 5.35
    bat_out_x = 6.35
    ret_x = 7.25

    # Pin 8 (VCC) -> +5V Battery -> Right Return Bus -> Pin 5 (GND)
    d.add(elm.Line().at(p8_ext).to((vcc_bus_x, y_p1)))
    d.add(elm.Dot().at((vcc_bus_x, y_p1)))
    d.add(elm.BatteryCell().at((bat_out_x, y_p1)).to((vcc_bus_x, y_p1)).label("5V", loc="top", fontsize=8.5))
    d.add(elm.Line().at((bat_out_x, y_p1)).to((ret_x, y_p1)))
    d.add(elm.Line().at((ret_x, y_p1)).to((ret_x, y_p4)))
    d.add(elm.Line().at(p5_ext).to((ret_x, y_p4)))

    # Ground symbol on bottom return bus
    gnd_x = 4.95
    d.add(elm.Dot().at((gnd_x, y_p4)))
    d.add(elm.Ground().at((gnd_x, y_p4)))

    # Pin 7 (VE Enable) -> Switch S -> +5V vertical bus (or GND if L)
    sw_en_l = 3.35
    sw_en_r = 4.35
    d.add(elm.Line().at(p7_ext).to((sw_en_l, y_p2)))
    if enable_state == "H (+5V)":
        d.add(elm.Line().at((sw_en_l, y_p2)).to((vcc_bus_x, y_p2)))
        ax.text((sw_en_l + sw_en_r) / 2, y_p2 + 0.18, "S (Closed)", ha="center", va="bottom", fontsize=7.8)
    elif enable_state == "L (GND)":
        d.add(elm.Line().at((sw_en_l, y_p2)).to((sw_en_l, y_p4)))
        d.add(elm.Dot().at((sw_en_l, y_p4)))
        ax.text(sw_en_l + 0.12, (y_p2 + y_p4) / 2, "VE = L (GND)", ha="left", va="center", fontsize=7.5)
    else:
        d.add(elm.Switch(contacts=True).at((sw_en_l, y_p2)).to((sw_en_r, y_p2)).label("S (Open)", loc="top", fontsize=7.8))
        d.add(elm.Line().at((sw_en_r, y_p2)).to((vcc_bus_x, y_p2)))

    # Vertical +5V bus from Pin 8 rail down to Pull-up resistor level
    y_pull = -0.35
    d.add(elm.Line().at((vcc_bus_x, y_p1)).to((vcc_bus_x, y_pull)))

    # Pin 6 (VO) -> Branch to 220 Ohm + LED (down to GND) and 1 kOhm Pull-up (right to +5V bus)
    branch_x = 3.15
    d.add(elm.Line().at(p6_ext).to((branch_x, y_p3)))
    d.add(elm.Dot().at((branch_x, y_p3)))

    # 1 kOhm Pull-up resistor to +5V bus
    d.add(elm.Line().at((branch_x, y_p3)).to((branch_x, y_pull)))
    d.add(
        elm.Resistor()
        .at((branch_x, y_pull))
        .to((vcc_bus_x, y_pull))
        .label(f"{rpull_ohm/1000:.1f}k$\\Omega$" if rpull_ohm >= 1000 else f"{rpull_ohm:.0f}$\\Omega$", loc="top", fontsize=8.2)
    )

    # 220 Ohm + Output Indicator LED down to GND line
    y_led_mid = -0.98
    d.add(
        elm.Resistor()
        .at((branch_x, y_p3))
        .to((branch_x, y_led_mid))
        .label(f"{rled_ohm:.0f}$\\Omega$", loc="right", fontsize=8.0)
    )
    led_lbl = "LED (ON)" if led_on else "LED (OFF)"
    d.add(
        elm.LED()
        .at((branch_x, y_led_mid))
        .to((branch_x, y_p4))
        .label(led_lbl, loc="right", fontsize=8.0)
    )
    d.add(elm.Dot().at((branch_x, y_p4)))

    d.draw(show=False)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf