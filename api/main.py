from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class PingResponse(BaseModel):
    message: str

@app.post("/ping", response_model=PingResponse)
async def ping():
    return PingResponse(message="Hello, I am alive")

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")  # Ensure image is in RGB format
    return np.array(image)

def predict_func(model, img, target_size=(256, 256)):
    # Resize image
    img_resized = tf.image.resize(img, target_size)
    img_array = tf.expand_dims(img_resized, 0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_array = tf.convert_to_tensor(image, dtype=tf.float32)
    predictions, confidence = predict_func(MODEL, img_array)
    return {
        'class': predictions,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
