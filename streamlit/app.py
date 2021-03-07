import streamlit as st
import SessionState
from PIL import Image

st.title("Welcome to License Plate Detectortron")
st.header("Localize and transcribe the license plate from an image!")

session_state = SessionState.get(detect_button=False)

uploaded_image = st.file_uploader(label="Upload an image of a car", type=["png", "jpeg", "jpg"])

if not uploaded_image:
	st.warning("Please upload an image.")
	st.stop()
else:
	session_state.uploaded_image = uploaded_image.read()
	st.image(session_state.uploaded_image, use_column_width=True)
	detect_button = st.button("Detect")