# filepath: [sentiment_api.py](http://_vscodecontentref_/0)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # ← TAMBAH INI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import re
import os

app = FastAPI(title="Indonesian Sentiment Analysis API", 
              description="API untuk analisis sentimen bahasa Indonesia dengan dukungan bahasa gaul",
              version="1.0.0")

# ← TAMBAH CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variable untuk model (akan diload jika tersedia)
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Try to load IndoBERT sentiment model, fallback to enhanced keyword if failed"""
    global model, tokenizer, model_loaded
    
    # List model alternatif yang bisa dicoba - PRIORITAS MODEL SENTIMENT
    model_options = [
        "mdhugol/indonesia-bert-sentiment-classification",  # PRIORITAS: Model khusus sentiment
        "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",  # Alternative sentiment model
        "indolem/indobert-base-uncased",  # Fallback: Base model
    ]
    
    for model_name in model_options:
        try:
            print(f"🔄 Trying to load model: {model_name}")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer
            print(f"📥 Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✅ Tokenizer loaded successfully!")
            
            # Load model
            print(f"📥 Downloading model {model_name} (this may take a while)...")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("✅ Model loaded successfully!")
            
            model_loaded = True
            print(f"🎉 {model_name} ready for sentiment analysis!")
            return  # Exit jika berhasil
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue  # Coba model berikutnya
    
    # Jika semua model gagal
    print("❌ All models failed to load")
    print("🔄 Using enhanced keyword-based analysis instead")
    model_loaded = False

# Try to load model on startup
load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    model_name = "Unknown"
    if model_loaded and model is not None:
        model_name = model.config.name_or_path if hasattr(model.config, 'name_or_path') else "Indonesian BERT Model"
    
    return {
        "message": "Indonesian Sentiment Analysis API", 
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": model_loaded,
        "model_name": model_name if model_loaded else "Enhanced Keyword Analysis",
        "model_type": "🤖 AI Model" if model_loaded else "📝 Keyword Analysis",
        "status": "🎉 Ready!" if model_loaded else "📝 Keyword Ready!"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model_loaded,
        "model_type": "IndoBERTweet" if model_loaded else "Enhanced Keyword Analysis",
        "ready": True
    }

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
    
    # Debug info
    print(f"🔍 Analyzing: '{text}'")
    print(f"🔧 Normalized: '{normalized_text}'")
    print(f"🤖 Model loaded: {model_loaded}")
    
    # Coba gunakan IndoBERTweet jika tersedia
    if model_loaded and model is not None and tokenizer is not None:
        try:
            import torch
            print("🎯 Using IndoBERTweet model...")
            
            # Tokenize input
            inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            print(f"📊 Model prediction: {pred} (confidence: {confidence:.3f})")
            print(f"📊 Probabilities: {probabilities.numpy()}")
            
            # Dynamic mapping berdasarkan jumlah kelas model
            num_classes = logits.shape[1]
            print(f"🔢 Number of classes: {num_classes}")
            
            if num_classes == 3:
                # 3-class model: 0=negative, 1=neutral, 2=positive (standard)
                if pred == 2:  # positive
                    result = 5 if confidence > 0.8 else 4
                    print(f"✅ Result: {result} stars (Positive)")
                    return result
                elif pred == 1:  # neutral
                    result = 3
                    print(f"😐 Result: {result} stars (Neutral)")
                    return result
                else:  # negative (pred == 0)
                    result = 1 if confidence > 0.8 else 2
                    print(f"❌ Result: {result} stars (Negative)")
                    return result
            
            elif num_classes == 5:
                # 5-class model: direct mapping to stars
                result = pred + 1  # 0-4 -> 1-5 stars
                print(f"⭐ Result: {result} stars (Direct mapping)")
                return result
            
            else:
                # Binary or other models - convert to 1-5 scale
                if pred == 1:  # positive (binary)
                    result = 5 if confidence > 0.8 else 4
                    print(f"✅ Result: {result} stars (Positive)")
                    return result
                else:  # negative (binary)
                    result = 1 if confidence > 0.8 else 2
                    print(f"❌ Result: {result} stars (Negative)")
                    return result
                
        except Exception as e:
            print(f"⚠️ Error using IndoBERTweet: {e}")
            print("🔄 Falling back to keyword analysis...")
    
    # Enhanced keyword-based analysis (fallback)
    print("🔤 Using enhanced keyword analysis...")
    result = enhanced_keyword_analysis(normalized_text, text)
    print(f"📝 Keyword analysis result: {result} stars")
    return result

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