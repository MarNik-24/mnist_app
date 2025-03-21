import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("About the Project")
st.markdown("""
### Classifying Handwritten Digits with Machine Learning

Classifying Handwritten Digits with Machine Learning
This project applies a machine learning approach to identify handwritten digits (0–9) using the mnist_784 dataset from OpenML.org. It combines k-NN and Extra Trees into an ensemble model for improved accuracy.
Users can interactively draw a digit on a canvas. The input goes through several preprocessing steps—such as grayscale conversion, resizing to 28×28 pixels, inversion, normalization, and flattening—before the model makes a prediction.

""")
