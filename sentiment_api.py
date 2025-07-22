# filepath: [sentiment_api.py](http://_vscodecontentref_/0)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

class TextRequest(BaseModel):
    text: str

def analyze_sentiment(text):
    # Ganti dengan model/analisis aslimu
    text = text.lower()
    if "senang" in text or "bahagia" in text or "happy" in text:
        return 5
    elif "marah" in text or "kesal" in text:
        return 2
    elif "sedih" in text or "down" in text:
        return 1
    else:
        return 3

@app.post("/predict")
async def predict(req: TextRequest):
    stars = analyze_sentiment(req.text)
    return JSONResponse(content={"stars": stars})