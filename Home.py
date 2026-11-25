import streamlit as st

st.set_page_config(
    page_title="Linear Integrated Circuits Virtual Lab",
    page_icon="⚡",
    layout="wide"
)

st.title("Linear Integrated Circuits Virtual Laboratory")
st.markdown(
    """
    Interactive simulation suite for **$\mu$A741 Operational Amplifier** and **IC 555 Timer** experiments, 
    designed according to *Op-Amps and Linear Integrated Circuits* by **Ramakant A. Gayakwad**.
    """
)
st.divider()

st.subheader("Experiment Index")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Part A: Op-Amp ($\mu$A741) Applications")
    st.page_link("pages/01_OpAmp_Integrator.py", label="1. Op-Amp Practical Integrator (Sine & Square Wave)", icon="📈")
    st.caption("Upcoming modules (2–11, 14–16): Differentiator, Square/Triangular/Sawtooth Generators, Wien Bridge, Schmitt Trigger, RC Phase Shift, Butterworth Filters, I-V & V-I Converters, Asymmetric Astable.")

with col2:
    st.markdown("#### Part B: IC 555 Timer Applications")
    st.caption("Upcoming modules (12–13, 17–18): 555 Astable Multivibrator, Monostable Multivibrator, Voltage Controlled Oscillator (VCO), and 555 Schmitt Trigger.")

st.divider()
st.info("💡 **Tip:** Use either the index links above or the left navigation sidebar to switch between experiments. Adjust component values in real time to observe CRO waveforms and frequency responses.")