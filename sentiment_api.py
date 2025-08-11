# filepath: [sentiment_api.py](http://_vscodecontentref_/0)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = FastAPI()

# Load IndoBERTweet Sentiment model & tokenizer
print("Loading IndoBERTweet model...")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobertweet-sentiment-classifier")
model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobertweet-sentiment-classifier")
print("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

def normalize_slang(text):
    """Normalisasi kata gaul/slang ke bahasa baku"""
    slang_dict = {
        'gw': 'saya', 'gue': 'saya', 'w': 'saya',
        'lu': 'kamu', 'elu': 'kamu', 'lo': 'kamu',
        'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'engga': 'tidak',
        'bgt': 'banget', 'bgt': 'sangat',
        'btw': 'ngomong ngomong', 'fyi': 'informasi',
        'yg': 'yang', 'yng': 'yang',
        'dgn': 'dengan', 'dg': 'dengan',
        'org': 'orang', 'orng': 'orang',
        'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah',
        'blm': 'belum', 'blom': 'belum',
        'bkn': 'bukan', 'bukan': 'bukan',
        'krn': 'karena', 'krna': 'karena',
        'trs': 'terus', 'trus': 'terus',
        'jg': 'juga', 'jga': 'juga',
        'aja': 'saja', 'ajah': 'saja',
        'emg': 'memang', 'emang': 'memang',
        'tp': 'tapi', 'tapi': 'tetapi',
        'kalo': 'kalau', 'klo': 'kalau',
        'gimana': 'bagaimana', 'gmn': 'bagaimana',
        'knp': 'kenapa', 'knapa': 'kenapa',
        'mantap': 'bagus', 'mantul': 'bagus',
        'anjay': 'wah', 'anjir': 'wah',
        'gabut': 'tidak ada kegiatan',
        'mager': 'malas gerak',
        'baper': 'bawa perasaan',
        'santuy': 'santai',
        'kepo': 'ingin tahu',
        'php': 'pemberi harapan palsu',
        'bucin': 'budak cinta'
    }
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace slang words
    for slang, formal in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', formal, text)
    
    return text

def analyze_sentiment(text):
    """Analisis sentimen menggunakan IndoBERTweet"""
    try:
        # Normalisasi kata gaul
        normalized_text = normalize_slang(text)
        
        # Tokenisasi dan prediksi
        inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
        
        # Mapping label IndoBERTweet ke rating bintang
        # IndoBERTweet: 0=negative, 1=neutral, 2=positive
        if pred == 2:  # positive
            return 5
        elif pred == 1:  # neutral
            return 3
        else:  # negative (pred == 0)
            return 1
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Fallback ke deteksi keyword sederhana
        text_lower = text.lower()
        if any(word in text_lower for word in ["senang", "bahagia", "happy", "mantap", "bagus", "keren", "suka"]):
            return 5
        elif any(word in text_lower for word in ["marah", "kesal", "benci", "jelek", "buruk", "anjir"]):
            return 2
        elif any(word in text_lower for word in ["sedih", "down", "galau", "baper"]):
            return 1
        else:
            return 3

@app.post("/predict")
async def predict(req: TextRequest):
    stars = analyze_sentiment(req.text)
    return JSONResponse(content={"stars": stars})