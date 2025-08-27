---
title: Indonesian Sentiment Analysis
emoji: 🎭
colorFrom: blue
colorTo: green
sdk: fastapi
sdk_version: 0.115.13
app_file: sentiment_api.py
pinned: false
license: mit
---

# Indonesian Sentiment Analysis API

API untuk analisis sentimen bahasa Indonesia dengan dukungan bahasa gaul menggunakan IndoBERT.

## Features
- Analisis sentimen bahasa Indonesia formal & informal
- Normalisasi 30+ kata gaul
- Model IndoBERT untuk akurasi tinggi
- Enhanced keyword analysis sebagai fallback

## Usage
- POST /predict dengan JSON: {"text": "teks yang ingin dianalisis"}
- Response: {"stars": 1-5}
