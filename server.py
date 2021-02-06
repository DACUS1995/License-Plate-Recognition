import json
from typing import List

from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from predict_classification import get_classification_prediction

device = torch.device(Config.device)
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	if file is None:
		raise HTTPException(status_code=404, detail="No file detected!")

	file_bytes = await file.read()
	class_id, class_name = get_classification_prediction(raw_input=file_bytes)
	return jsonify({"class_id": class_id, "class_name": class_name})

@app.get("/")
async def root():
	return "Hello"