import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import sys

from Utils.TextConvert import embed_txt
from Utils.ImageSearch import imageSearch
from Utils.rotateImage import rotate_image
from trajectoryGraph import graphTraj
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Find and Display"
)

# Page Title
st.title("Lorem ipsum dolor!")

# Create a form
with st.form("prompt"):
    st.write("Prompt goes here:")
    Prompt = st.text_input("Prompt")
    # Every form must have a submit button
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        embbededText = embed_txt(Prompt)
        imag = imageSearch(embbededText)
        image_path_str = str(imag.image_path.iloc[0])
        out = rotate_image(Image.open(image_path_str))
        # Use columns to place the image and graph side by side
        col1, col2 = st.columns(2)

        trajectory = imag['trajectory_data'].array[0].split()

        print(trajectory)

        imag_x = float(trajectory[-3])
        imag_y = float(trajectory[-2])
        
        with col1:
            st.image(out, width = 330, caption= image_path_str + ", an image of " + Prompt)
        
        with col2:
            graphTraj(imag_x, imag_y)  # 0,0 is a placeholder and will be replaced
