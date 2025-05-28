from fastapi import FastAPI, File, UploadFile
from model import ONNXModel, ImagePreProcessor
import numpy as np
import uvicorn
from PIL import Image
import io

app = FastAPI()

model = ONNXModel()
preprocessor = ImagePreProcessor()

@app.get("/")
def read_root():
    return {"status": "Model is up and running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_array = preprocessor.transform(image).unsqueeze(0).numpy()
    output = model.predict(input_array)
    predicted_class = int(np.argmax(output))
    return {"predicted_class": predicted_class}