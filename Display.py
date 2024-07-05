import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import sys

from Utils.TextConvert import embed_txt
from trajectoryGraph import graphTraj

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
        embbededText = TextConvert.embed_txt(Prompt)
        imag = ImageSearch.imageSearch(embbededText)
        image_path_str = str(imag.image_path.iloc[0])
        out = rotateImage.rotate_image(Image.open(image_path_str))
        # Use columns to place the image and graph side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(out, caption= out + ", an image of " + Prompt)
        
        with col2:
            graphTraj(0, 0)  # 0,0 is a placeholder and will be replaced
