import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Mapping of class indices to disease names
class_to_disease = {
    0: "Cassava Brown Streak Disease",
    1: "Cassava Green Mottle",
    2: "Cassava Healthy",
    
    3: "Cassava Mosaic Disease",
}

# Page Configurations
st.set_page_config(
    page_title="Cassava Disease Classification App",
    page_icon="🌿",
    layout="centered",
)

# CSS Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
            color: #495057;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            max-width: 800px;
        }
        .stButton {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .stButton:hover {
            background-color: #218838;
        }
        .stInput {
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 10px;
            width: 100%;
            margin-bottom: 15px;
        }
        .stOutput {
            padding: 20px;
            border-radius: 4px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .stFooter {
            margin-top: 30px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("Cassava Disease Classification App")

# Upload Image Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Predict Button
    if st.button("Predict", key="predict_button"):
        # Preprocess the image for model prediction
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        disease_name = class_to_disease.get(class_index, "Unknown Disease")

        # Display Prediction
        st.success(f"Predicted Disease: {disease_name}")

# Footer Section
st.markdown(
    """
    <div class="stFooter">
        <p>Powered by Streamlit and TensorFlow</p>
    </div>
    """,
    unsafe_allow_html=True
)
