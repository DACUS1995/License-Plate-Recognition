import streamlit as st
import SessionState
from PIL import Image
from detector import Detector

st.title("Welcome to License Plate Detectortron")
st.header("Localize and transcribe the license plate from an image!")

def detect(image) -> str:
	detector = Detector.getInstance()
	image = Image.open(io.BytesIO(file_bytes))
	image = detector.preprocess_image(image)
	license_plate = detector.detect_single(image=image, method="tesseract")
	return license_plate

session_state = SessionState.get(detect_button=False)
pred_button = None
uploaded_image = st.file_uploader(label="Upload an image of a car", type=["png", "jpeg", "jpg"])

if not uploaded_image:
	st.warning("Please upload an image.")
	st.stop()
else:
	session_state.uploaded_image = uploaded_image.read()
	st.image(session_state.uploaded_image, use_column_width=True)
	detect_button = st.button("Detect")

if pred_button:
	session_state.pred_button = True 

if session_state.pred_button:
	session_state.pred_license = detect(session_state.uploaded_image)
	st.write(f"License plate: {session_state.pred_license}")