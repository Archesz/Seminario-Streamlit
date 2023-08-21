import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import requests

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras import layers
from keras.optimizers import Adam, RMSprop
import keras

def set_background(image_url, height):
    response = requests.get(image_url)
    if response.status_code == 200:
        img_data = response.content
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: 100vw {height}px; /* Keep aspect ratio, set height to 200px */
                background-repeat: no-repeat;
                }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.write("Falha em carregar a imagem.")

def classify(image, model, class_names):
    # Converter a imagem no tamanho (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Converter em um arquivo Numpy
    image_array = np.asarray(image)

    # Normalizar
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Definir entrada do modeo
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predição
    prediction = model.predict(data)

    # Revebendo saida
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    
    # Score
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def createModel(shape, include_top, num_classes, layers_list):

    IMG_SIZE = (shape,shape)

    vgg = VGG16(
    include_top=include_top,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
    )

    NUM_CLASSES = num_classes

    vgg16 = Sequential()
    vgg16.add(vgg)

    for layer in layers_list:
        if layer == "Dropout":
            vgg16.add(layers.Dropout(0.3))
        elif layer == "Flatten":
            vgg16.add(layers.Flatten())
            vgg16.add(layers.Dropout(0.5))
        elif layer == "Dense":
            vgg16.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    vgg16.layers[0].trainable = False

    vgg16.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    vgg16.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

    return vgg16