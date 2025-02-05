import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

@st.cache_data()
def load():
    model_path = "best_model_recyclage.keras"
    model = load_model(model_path, compile=False)
    return model

# Chargement du modèle
model = load()

def predict(upload):
    image = Image.open(upload)
    image = np.asarray(image)
    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)

    prediction = model.predict(image_resized)

    rec = prediction[0][0]    

    return rec

st.title('Poubelle Intelligente')

upload = st.file_uploader('Chargez l''image de votre objet', type = ['jpg', 'jpeg', 'png'])

c1, c2 = st.columns(2)

if upload:
    rec = predict(upload)
    prob_recyclable = rec * 100      
    prob_organic = (1-rec)*100

    c1.image(Image.open(upload))
    if prob_recyclable > 50:
        c2.write(f"Je suis certain à {prob_recyclable:.2f} % que l'objet est recyclable")
    else:
        c2.write(f"Je suis certain à {prob_organic:.2f} % que l'objet n'est pas recyclable")

