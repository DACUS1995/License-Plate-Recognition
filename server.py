import json
from typing import List

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from detector import Detector

app = FastAPI()

origins = [
	"http://localhost:8080"
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
	if file is None:
		raise HTTPException(status_code=404, detail="No image detected!")

	file_bytes = await file.read()
	detector = Detector.getInstance()

	image = Image.open(io.BytesIO(file_bytes))
	image = detector.preprocess_image(image)
	license_plate = detector.detect_single(image=image)

	return jsonable_encoder({"license": license_plate})

@app.get("/")
async def root():
	return "Hello"