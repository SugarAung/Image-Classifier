import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model = tf.keras.models.load_model('sset_model.h5')
labels = ['Egg Tart', 'Salmon Sashimi', 'Unknown']

def predict_image(image):
    image = ImageOps.fit(image, (75, 75), Image.LANCZOS)
    image = np.asarray(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    confidence = np.max(prediction)
    label_index = np.argmax(prediction)

    if confidence < 0.7:
        return "Unknown (low confidence)", prediction[0]
    return f"{labels[label_index]} ({confidence:.2f})", prediction[0]

st.title(" Egg Tart vs Salmon Sashimi vs Unknown")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    label, probs = predict_image(img)
    st.success(f"Prediction: {label}")
    st.write({labels[i]: float(probs[i]) for i in range(3)})