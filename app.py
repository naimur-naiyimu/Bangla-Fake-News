import gradio as gr
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import os

# 1. Load using absolute path
abs_model_path = os.path.abspath('Models/bangla-bert-base')
try:
    model = tf.keras.models.load_model(os.path.join(abs_model_path, 'bangla_bert_model.keras'))
    tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
    print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Loading failed: {e}")

def predict_news(text):
    if not text.strip(): return "Please enter text."
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='np')
    outputs = model(inputs.data)
    # Adjusting for potential Keras model output vs Transformers wrapper output
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    prediction = np.argmax(logits, axis=1)[0]
    return 'Authentic 🇧🇩' if prediction == 1 else 'Fake 🚫'

iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=5, label="Bangla News Content"),
    outputs=gr.Label(label="Prediction"),
    title="Bangla Fake News Detector"
)

iface.launch()