#!/usr/bin/env python3
"""Test script untuk sentiment analysis"""

from sentiment_api import analyze_sentiment

# Test cases
test_cases = [
    "gue seneng banget hari ini",      # Should be 5 (very positive)
    "capek banget, stress",            # Should be 1-2 (negative)  
    "biasa aja sih",                   # Should be 3 (neutral)
    "anjay mantul banget",             # Should be 4-5 (positive)
    "sedih banget hari ini"            # Should be 1-2 (negative)
]

print("🧪 Testing Sentiment Analysis...")
print("=" * 50)

for i, text in enumerate(test_cases, 1):
    print(f"\n{i}. Text: '{text}'")
    result = analyze_sentiment(text)
    print(f"   Result: {result} stars")
    
    # Expected vs actual
    if "seneng banget" in text or "mantul banget" in text:
        expected = "4-5 stars (positive)"
    elif "capek" in text and "stress" in text or "sedih banget" in text:
        expected = "1-2 stars (negative)"
    elif "biasa aja" in text:
        expected = "3 stars (neutral)"
    else:
        expected = "unknown"
    
    print(f"   Expected: {expected}")
    print("-" * 30)

print("\n✅ Test completed!")
