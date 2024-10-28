import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Downloads/pneumonia_classifier.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (150, 150))  # Resize to match model input
    image = image.astype('float32') / 255.0  # Normalize
    return np.expand_dims(image, axis=-1)  # Add channel dimension

# Streamlit app layout
st.title("Pneumonia Detection Model")
st.write("Upload a chest X-ray image:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    prediction_class = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

    st.write(f"Prediction: {prediction_class}")