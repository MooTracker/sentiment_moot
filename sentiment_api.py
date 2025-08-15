# filepath: [sentiment_api.py](http://_vscodecontentref_/0)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import re

app = FastAPI()

# Global variable untuk model (akan diload jika tersedia)
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Try to load IndoBERTweet, fallback to enhanced keyword if failed"""
    global model, tokenizer, model_loaded
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Trying to load IndoBERTweet model...")
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobertweet-sentiment-classifier")
        model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobertweet-sentiment-classifier")
        model_loaded = True
        print("IndoBERTweet model loaded successfully!")
    except Exception as e:
        print(f"Failed to load IndoBERTweet: {e}")
        print("Using enhanced keyword-based analysis instead")
        model_loaded = False

# Try to load model on startup
load_model()

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
        'bucin': 'budak cinta',
        # Tambahan kata positif yang sering dipakai
        'seneng': 'senang', 'senang': 'senang',
        'bahagia': 'bahagia', 'happy': 'senang',
        'kamaren': 'kemarin', 'kemaren': 'kemarin'
    }
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace slang words
    for slang, formal in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', formal, text)
    
    return text

def analyze_sentiment(text):
    """Analisis sentimen dengan IndoBERTweet atau enhanced keyword"""
    global model, tokenizer, model_loaded
    
    # Normalisasi kata gaul
    normalized_text = normalize_slang(text)
    
    # Coba gunakan IndoBERTweet jika tersedia
    if model_loaded and model is not None:
        try:
            import torch
            inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            
            # Mapping label IndoBERTweet ke rating bintang
            if pred == 2:  # positive
                return 5
            elif pred == 1:  # neutral
                return 3
            else:  # negative
                return 1
                
        except Exception as e:
            print(f"Error using IndoBERTweet: {e}")
            # Fallback ke enhanced keyword
    
    # Enhanced keyword-based analysis (fallback)
    return enhanced_keyword_analysis(normalized_text, text)

def enhanced_keyword_analysis(normalized_text, original_text):
    """Enhanced keyword analysis untuk bahasa Indonesia + slang"""
    text_lower = normalized_text.lower()
    
    # Positive keywords (diperbanyak dan lebih sensitif)
    positive_words = [
        "senang", "bahagia", "happy", "mantap", "bagus", "keren", "suka", "cinta", "love",
        "amazing", "luar biasa", "hebat", "fantastis", "sempurna", "excellent", "good",
        "positif", "optimis", "gembiraan", "kebahagiaan", "sukses", "berhasil", "menang",
        "excited", "antusias", "semangat", "motivasi", "inspirasi", "grateful", "bersyukur",
        "mantul", "jos", "top", "juara", "recommended", "worth it", "puas", "satisfied",
        "gembira", "asyik", "asik", "cool", "nice", "wonderful", "great", "awesome"
    ]
    
    # Strong positive words (kata yang sangat positif)
    strong_positive_words = [
        "banget", "sangat", "luar biasa", "fantastis", "sempurna", "amazing", "awesome",
        "gembira", "bahagia banget", "senang banget", "happy banget"
    ]
    
    # Negative keywords (diperbanyak)
    negative_words = [
        "marah", "kesal", "benci", "jelek", "buruk", "jahat", "sedih", "kecewa", "galau",
        "frustrated", "angry", "hate", "bad", "terrible", "awful", "horrible", "disgusting",
        "menyebalkan", "annoying", "stress", "depresi", "down", "hopeless", "putus asa",
        "fail", "gagal", "rugi", "loss", "disappointed", "broken heart", "sakit hati",
        "toxic", "drama", "problem", "masalah", "susah", "sulit", "capek", "tired"
    ]
    
    # Neutral/Mixed keywords
    neutral_words = [
        "biasa", "standard", "normal", "okay", "ok", "fine", "lumayan", "so so",
        "average", "medium", "moderate", "netral", "balanced", "mixed feelings"
    ]
    
    # Negation words (kata negasi)
    negation_words = ["tidak", "bukan", "jangan", "gak", "ga", "engga", "no", "nope", "never"]
    
    # Count sentiment words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    strong_positive_count = sum(1 for word in strong_positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    neutral_count = sum(1 for word in neutral_words if word in text_lower)
    
    # Check for combinations like "senang banget"
    if "senang banget" in text_lower or "bahagia banget" in text_lower or "happy banget" in text_lower:
        strong_positive_count += 2
    
    # Check for negations
    has_negation = any(neg in text_lower for neg in negation_words)
    
    # Advanced scoring with context
    if has_negation:
        # If there's negation, flip the sentiment partially
        if positive_count > negative_count:
            return 3  # Neutral instead of positive
        elif negative_count > positive_count:
            return 4  # Less negative
    
    # Calculate sentiment score with strong positive bonus
    total_positive = positive_count + (strong_positive_count * 2)  # Strong words worth double
    
    if total_positive > negative_count + neutral_count:
        return 5  # Strong positive
    elif total_positive > negative_count:
        return 4  # Mild positive
    elif negative_count > total_positive + neutral_count:
        return 1  # Strong negative
    elif negative_count > total_positive:
        return 2  # Mild negative
    else:
        return 3  # Neutral

@app.post("/predict")
async def predict(req: TextRequest):
    stars = analyze_sentiment(req.text)
    return JSONResponse(content={"stars": stars})