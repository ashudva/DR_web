import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from pathlib import Path
import pandas as pd

model_dir = Path("models/base.h5")
model = keras.models.load_model(model_dir)

st.title("Handwritten Digit Recognition")
st.text("Draw a digit!")
st.text(" ")
st.text(" ")
st.text(" ")

col1, col2 = st.beta_columns(2)
with col1:
    mode = st.checkbox("Draw or Delete", True)
    # define size of the canvas
    SIZE = 192
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas')

with col2:
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(
            img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Input Image for the model')
        st.image(rescaled)

if st.button('Predict the Digit'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    test_x = test_x.reshape(1, 28, 28, 1) / 255
    val = model.predict(test_x)
    val = np.around(val, 3)
    st.write(f'Predicted Digit: {np.argmax(val)}')
    st.write("Prediction Probabilities")
    st.write(np.around(val, 4))
    st.bar_chart(val.reshape(10, 1))
