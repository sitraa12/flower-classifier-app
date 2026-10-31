import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Flower Classifier")
st.write("Upload a flower image and let the model predict its type.")

@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model("flower_classifier.h5", compile=False)

model = load_flower_model()

class_names = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded flower", use_container_width=True)

    image_resized = image.resize((160, 160))
    image_array = np.array(image_resized, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    st.success(
        f"Predicted flower: {class_names[predicted_index].title()}"
    )

    st.write(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

    st.subheader("Prediction scores")

    st.bar_chart({
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    })