# Useful Importzzzz...
from keras.backend_config import image_data_format
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import cv2
from PIL import Image, ImageOps

# Loading the tensorflow model.
model = keras.models.load_model("model/potato_disease_clf.h5")

# Setting the page congifuration.
icon = "bg.png"
st.set_page_config(page_title="Potato Disease Classification", page_icon=icon)

# Function which helps to predict and display resullts on streamlit.
def predictions_of_model(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    predictions = model.predict(img_reshape)
    return predictions

# Hide Main Manu and footer note from streamlit.
st.markdown("""
    <style> 
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Defining the main function.
def main():

    # Common Bootstrap CDN.
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    # Now streamlit GUI script starts from here.
    st.markdown("""
        <nav class="navbar fixed-top navbar-expand-lg navbar-light" style="background-color: #F9F1F0;">
        <a class="navbar-brand" href="https://www.linkedin.com/in/bhavik-jadav-5a281b1b4/" target="_blank">We4Web Solutions - Bhavik Jadav</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="https://www.instagram.com/_bhaviik__/" target="_blank">Instagram</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="https://twitter.com/BhavikJ83561708" target="_blank">Twitter</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="https://github.com/bhavikjadav" target="_blank">GitHub</a>
            </li>
            </ul>
        </div>
        </nav>""", unsafe_allow_html=True)

    # TItle of the app.
    st.write("### **Potato Disease Classification** *Using Convolutional Neural Network*.")

    # Description About Project.
    st.markdown("""In potato farming, farmers suffer a lot if the potatoes are spoiled.
    What happens if you look at its leaves and find out about future diseases? 
    We can prevent this from happening with the help of Deep Learning Techniques.
    To achieve this goal I've used Convolutional Neural Netwoek Architecture which helps use 
    to find patterns in various images.""")
    
    st.markdown("""Solution:\n
    Step 1 :
    We will click the image of the potato leave on a mobile phone or any other device.
    
    Step 2 : 
    We will then cllick the photo or upload the leave image from device.
    
    Step 3 : 
    Then our deep learning model will predict whether the potato is suffering from
    Disease or Not.""")

    # Predcting the Result and Printing the Score.
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    uploaded_file = st.file_uploader("Upload Files",type=['png','JPG'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        prediction = predictions_of_model(image, model)
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            st.markdown("""### Predicted Result : **Early Blight Disease**""")
        elif predicted_class == 1:
            st.markdown("""### Predicted Result : **Late Blight Disease**""")
        elif predicted_class == 2:
            st.markdown("""### Predicted Result : **Healthy**""")
        confidence = round(100 * (np.argmax(prediction)), 2)
        print(f"Confidence : " + str(confidence) + "%")
        # print(prediction)

if __name__ == "__main__":
    main()
