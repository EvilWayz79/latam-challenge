import fastapi
import requests
from model import DelayModel
import pandas as pd

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    model = DelayModel()
    data = pd.DataFrame(requests.data)
    features, _ = model.preprocess(data)
    predictions = model.predict(features)
    return {"predictions": predictions}