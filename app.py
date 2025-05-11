import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os

# Define label dictionary
dic = {0: 'Grade 0 : Normal', 1: 'Grade 1 : Doubtful', 2: 'Grade 2 : Mild', 3: 'Grade 3 : Moderate', 4: 'Grade 4 : Severe'}

# Image size
img_size = 256

# Set the filepath variable
filepath = 'model2.keras'  # Path to your model file

# Load model using the filepath
model = load_model(filepath)

# Define predict function
def predict_label(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]

# Streamlit app
st.set_page_config(page_title="Osteoarthritis", layout="centered")

st.title("Diagnosis for the Prediction of Knee Osteoarthritis Using Deep Learning")
st.write("Diagnosis for the Prediction of Knee Osteoarthritis Using Deep Learning. Choose your Knee X-Ray visual file and click Predict to get your diagnosis.")
st.write("Upload an image and get the prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Show the uploaded image
    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        prediction = predict_label(img)
        st.success(f"Prediction: **{prediction}**")
