from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib


app = FastAPI(title = 'Gold Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Reemplaza con el origen de tu aplicación Flutter
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load(pathlib.Path('model/model_ps.joblib'))

class InputData(BaseModel):
    year:int = 2016
    mileage: int = 40145
    city:float =1769.0
    state:float = 44.0
    make:float  =15.0

class OutputData(BaseModel):
    price:float=0.8

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)

    return {'price':result}