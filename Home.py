import os
import streamlit as st
from utils import render_sidebar_header

st.set_page_config(
    page_title="Linear Integrated Circuits Virtual Lab",
    page_icon="⚡",
    layout="wide"
)
render_sidebar_header()

st.title("Linear Integrated Circuits Virtual Laboratory")
st.markdown(
    """
    Interactive simulation suite for **$\mu$A741 Operational Amplifier** and **IC 555 Timer** experiments, 
    designed according to *Op-Amps and Linear Integrated Circuits* by **Ramakant A. Gayakwad**.
    """
)
st.divider()

# Complete master catalog of all 18 experiments
OPAMP_EXPERIMENTS = [
    ("pages/01_OpAmp_Integrator.py", "1. Practical Integrator using Op-Amp (µA741)", "📈"),
    ("pages/02_OpAmp_Differentiator.py", "2. Practical Differentiator using Op-Amp (µA741)", "📉"),
    ("pages/03_Square_Wave_Generator.py", "3. Square Wave Generator using Op-Amp (µA741)", "🔲"),
    ("pages/04_Triangular_Wave_Generator.py", "4. Triangular Wave Generator using Op-Amp (µA741)", "🔺"),
    ("pages/05_Sawtooth_Wave_Generator.py", "5. Sawtooth Wave Generator using Op-Amp (µA741)", "📐"),
    ("pages/06_Wien_Bridge_Oscillator.py", "6. Wien Bridge Oscillator with Amplitude Stabilization", "〰️"),
    ("pages/07_Schmitt_Trigger_Hysteresis.py", "7. Schmitt Trigger & Hysteresis Curve (µA741)", "🔁"),
    ("pages/08_RC_Phase_Shift_Oscillator.py", "8. RC Phase Shift Oscillator using Op-Amp (µA741)", "🌀"),
    ("pages/09_First_Order_Low_Pass_Butterworth_Filter.py", "9. First Order Low-Pass Butterworth Filter", "🔽"),
    ("pages/10_Second_Order_Low_Pass_Butterworth_Filter.py", "10. Second Order Low-Pass Butterworth Filter", "🔽"),
    ("pages/11_First_Order_High_Pass_Butterworth_Filter.py", "11. First Order High-Pass Butterworth Filter", "🔼"),
    ("pages/12_Second_Order_High_Pass_Butterworth_Filter.py", "12. Second Order High-Pass Butterworth Filter", "🔼"),
    ("pages/13_Narrow_Band_Pass_Filter.py", "13. First Order Narrow Band-Pass Butterworth Filter", "📶"),
    ("pages/14_Current_to_Voltage_Converter.py", "14. Current-to-Voltage Converter (µA741)", "🔌"),
    ("pages/15_Voltage_to_Current_Converter.py", "15. Voltage-to-Current Converter (µA741)", "🔋"),
]

TIMER_555_EXPERIMENTS = [
    ("pages/16_555_Astable_Multivibrator.py", "16. Astable Multivibrator using IC 555 Timer", "⏲️"),
    ("pages/17_555_Monostable_Multivibrator.py", "17. Monostable Multivibrator using IC 555 Timer", "1️⃣"),
    ("pages/18_555_Voltage_Controlled_Oscillator.py", "18. Voltage Controlled Oscillator (VCO) using IC 555", "🎛️"),
    ("pages/19_555_Schmitt_Trigger.py", "19. Schmitt Trigger Circuit using IC 555 Timer", "🔀"),
]

def render_experiment_list(exp_list):
    for filepath, label, icon in exp_list:
        if os.path.exists(filepath):
            st.page_link(filepath, label=label, icon=icon)
        else:
            st.markdown(f"<span style='color:#64748b;'>{icon} {label} *(Module pending)*</span>", unsafe_allow_html=True)

st.subheader("Experiment Index")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Part A: Op-Amp ($\mu$A741) Applications")
    render_experiment_list(OPAMP_EXPERIMENTS)

with col2:
    st.markdown("#### Part B: IC 555 Timer Applications")
    render_experiment_list(TIMER_555_EXPERIMENTS)

st.divider()
st.caption(
    "Developed by **Dr. Nibu A George**, Associate Professor of Physics, Baselius College (Autonomous), Kottayam | "
    "Reference: R. A. Gayakwad, *Op-Amps and Linear Integrated Circuits*."
)
