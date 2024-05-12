import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

model = load_model()

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Names", "About"])

# Home Page
if page == "Home":
    st.title("Brain Tumor MRI Classification")
    st.markdown("---")

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
        result = class_names[np.argmax(prediction)]
        if result == 'No Tumor':
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

# Names Page
elif page == "Group Members":
    st.title("Names")
    st.markdown("---")
    st.write("Here are the members of GROUP 3")
    st.write("- Ejercito, Marlon")
    st.write("- Flores, Joshua")
    st.write("- Flores, Marc")
    st.write("- Gabiano, Leonard")
    st.write("- Gomez, Joram")

# About Page
elif page == "About":
    st.title("About")
    st.markdown("---")
    st.write("This is a simple web application that classifies Brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor.")
    st.write("It uses a deep learning model trained on MRI images to make predictions.")
