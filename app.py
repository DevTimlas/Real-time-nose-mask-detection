import streamlit as st
from rtnmt1 import make_predict
from PIL import Image, ImageOps

st.title("Face Mask Detection with TF")
st.header("Face Mask Detection Test with Tensroflow")
st.text("Upload a face Image to detect maybe the person is wearing a face mask or not.")

uploaded_file = st.file_uploader("Choose a face Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded FaceImage.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = make_predict(image)
    if label == 0:
        st.write("The person is wearing a face mask")
    else:
        st.write("The person is not wearing a face mask")
