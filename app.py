import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

# Page Title and Description
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image of a brain and classify whether it has Glioma, Meningioma, No Tumor, or Pituitary Tumor.")

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

model = load_model()

# File Uploader
file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

# Function to make predictions
def import_and_predict(image_data, model):
    size = (150, 150)  
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS) 
    img = np.asarray(image)
    img = img / 255.0  
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Display the results
if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = import_and_predict(image, model)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    string = "Prediction: " + class_names[np.argmax(prediction)]
    st.success(string)
