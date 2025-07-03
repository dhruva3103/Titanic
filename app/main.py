from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.schema import PassengerInput
from app.model import predict_passenger, train_and_evaluate, load_data
from scripts.registry import list_registered_models
from scripts.ingest import ingest_data

app = FastAPI(title="Titanic Survival Prediction API")

@app.get("/")
def read_root():
    return {"status": "Titanic Prediction API is running"}

@app.post("/predict")
def predict(input: PassengerInput):
    result = predict_passenger(input.dict())
    return {
        "prediction": result,
        "label": "Survived" if result == 1 else "Did not survive"
    }

@app.post("/train")
def train(file: UploadFile = File(...)):
    # Save uploaded file and ingest it
    raw_path = 'train.csv'
    with open(raw_path, 'wb') as f:
        f.write(file.file.read())

    ingest_data(input_path=raw_path)
    df = pd.read_csv('data/processed/clean_titanic.csv')
    train_and_evaluate(df, target_column='Survived')
    return {"status": "Model trained, ingested, and registered successfully."}

@app.get("/registry")
def get_registry():
    return list_registered_models()
