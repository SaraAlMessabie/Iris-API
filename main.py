from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = joblib.load("iris_model.pkl")
@app.get("/")
def home():
    return {"message":"Welcome to AI API"}

from pydantic import BaseModel
class IrisInput(BaseModel):
    features: list[float]
    
@app.post("/predict")
def predict(data:IrisInput):
   prediction = model.predict([data.features])
   return {"prediction": int(prediction[0])}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    alow_methods=["*"],
    allow_headers=["*"]
)
