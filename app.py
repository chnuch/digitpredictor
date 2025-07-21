import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from keras.models import load_model


model = load_model("mnist_digit_model.keras")

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]  
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    st.write("### Predicted Digit:", np.argmax(prediction))
