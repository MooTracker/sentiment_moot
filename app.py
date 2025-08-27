import gradio as gr
import os

# Import dari file sentiment_api HANYA fungsi yang dibutuhkan
def predict_gradio(text):
    """Simplified prediction function untuk Gradio"""
    if not text.strip():
        return "❌ Mohon masukkan teks terlebih dahulu"
    
    try:
        # Import di dalam fungsi untuk menghindari conflicts
        from sentiment_api import analyze_sentiment
        stars = analyze_sentiment(text)
        
        # Convert stars ke emoji dan description
        if stars == 5:
            return f"⭐⭐⭐⭐⭐ ({stars}/5) - Sangat Positif!"
        elif stars == 4:
            return f"⭐⭐⭐⭐ ({stars}/5) - Positif"
        elif stars == 3:
            return f"⭐⭐⭐ ({stars}/5) - Netral"
        elif stars == 2:
            return f"⭐⭐ ({stars}/5) - Negatif"
        else:
            return f"⭐ ({stars}/5) - Sangat Negatif"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio interface - SIMPLIFIED
with gr.Blocks(title="🎭 Indonesian Sentiment Analysis") as demo:
    gr.Markdown("# 🎭 Indonesian Sentiment Analysis")
    gr.Markdown("""
    **API analisis sentimen bahasa Indonesia dengan dukungan bahasa gaul menggunakan IndoBERT**
    
    ✨ **Fitur:**
    - 🤖 Model AI untuk akurasi tinggi
    - 🗣️ Support bahasa gaul/slang Indonesia  
    - 📊 Rating 1-5 bintang
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Masukkan teks bahasa Indonesia", 
                placeholder="Contoh: gue seneng banget hari ini",
                lines=3
            )
            predict_btn = gr.Button("🔮 Analisis Sentimen", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="🎭 Hasil Analisis Sentimen", lines=2)
    
    # Examples
    gr.Examples(
        examples=[
            ["gue seneng banget hari ini"],
            ["capek bgt tapi gak stress"], 
            ["anjay mantul banget dah"],
            ["lagi gabut nih, bosen"],
            ["bahagia banget sama hasil ini"]
        ],
        inputs=input_text
    )
    
    # Event handler
    predict_btn.click(
        fn=predict_gradio,
        inputs=input_text,
        outputs=output_text
    )

# Launch dengan pengaturan yang kompatibel dengan HuggingFace Spaces
if __name__ == "__main__":
    # HuggingFace Spaces akan otomatis set port
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )