import streamlit as st

st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")
if st.checkbox("Main Checkbox"):
    st.text("Check Box Active")

slider_value = st.slider("Slider", min_value=0.5, max_value=3.5)
st.text(f"Slider value is {slider_value}")

st.sidebar.text("text on side panel")
st.sidebar.checkbox("Side Panel Checkbox")