# Indonesian Sentiment Analysis API

API untuk analisis sentimen bahasa Indonesia dengan dukungan bahasa gaul/slang.

## Features
- ✅ Analisis sentimen bahasa Indonesia formal & informal
- ✅ Normalisasi 30+ kata gaul (gw, bgt, anjay, dll)
- ✅ Enhanced keyword analysis dengan 100+ kata
- ✅ Deteksi negasi & konteks
- ✅ Fallback system yang robust

## Deployment Options

### 1. Railway (Recommended - Free)
1. Push code ke GitHub
2. Connect Railway ke repo GitHub
3. Deploy otomatis!
4. URL: `https://your-app.railway.app`

### 2. Render (Free tier)
1. Push code ke GitHub
2. Connect Render ke repo
3. Deploy dengan build command: `pip install -r requirements.txt`
4. Start command: `uvicorn sentiment_api:app --host 0.0.0.0 --port $PORT`

### 3. Heroku
```bash
heroku create your-sentiment-api
git add .
git commit -m "Deploy sentiment API"
git push heroku main
```

### 4. Always Data
1. Upload files via FTP
2. Configure Python environment
3. Set startup script

## API Endpoints

### POST /predict
```json
{
  "text": "gue seneng banget hari ini"
}
```

Response:
```json
{
  "stars": 5
}
```

### GET /health
Health check endpoint

### GET /docs
Interactive API documentation

## Local Development
```bash
pip install -r requirements.txt
uvicorn sentiment_api:app --reload --port 8000
```

## Examples
- `"gue seneng banget hari ini"` → 5 stars
- `"capek bgt tapi gk stress"` → 4 stars  
- `"gembira"` → 5 stars
- `"sedih banget"` → 1 star
