import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ✅ MUST BE HERE before any other st. function
st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

st.title("AI Image Caption Generator")

# Load models and tokenizer with caching to avoid reloading on every interaction
@st.cache_resource
def load_models():
    try:
        caption_model = load_model("model.keras")
        feature_extractor = load_model("feature_extractor.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return caption_model, feature_extractor, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None

caption_model, feature_extractor, tokenizer = load_models()

def generate_caption(image):
    if image is None:
        return "Please upload an image"

    try:
        # Preprocess image
        img = image.resize((224, 224)).convert("RGB")
        img_array = np.array(img) / 255.0
        if img_array.shape != (224, 224, 3):
            raise ValueError("Invalid image dimensions")
        
        img_input = np.expand_dims(img_array, axis=0)
        image_features = feature_extractor.predict(img_input, verbose=0)

        # Generate caption
        caption = "startseq"
        for _ in range(34):  # max_length used during training
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=34)
            yhat = caption_model.predict([image_features, seq], verbose=0)
            predicted_word = tokenizer.index_word.get(np.argmax(yhat), "")
            if not predicted_word or predicted_word == "endseq":
                break
            caption += " " + predicted_word

        return caption.replace("startseq", "").replace("endseq", "").strip()

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Streamlit UI
st.title("📸 AI Image Caption Generator")
st.write("Upload an image, and the AI will generate a descriptive caption.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
    st.success("Caption Generated:")
    st.write(f"📝 **{caption}**")
