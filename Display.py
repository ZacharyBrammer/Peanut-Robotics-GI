import altair as alt
from PIL import Image
from Utils import rotateImage
# import numpy as np
# import pandas as pd
from Utils import ImageSearch
from Utils import TextConvert
import streamlit as st
import sys
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
        st.write(Prompt)
        embbededText = TextConvert.embed_txt(Prompt)
        imag = ImageSearch.imageSearch(embbededText)
        image_path_str = str(imag.image_path.iloc[0])
        out = rotateImage.rotate_image(Image.open(image_path_str))
        # image_path_str = str(out.image_path.iloc[0])
        # out = rotateImage.rotate_image(Image.open(image_path_str))


        # print(out["image_path"])
        # Use columns to place the image and graph side by side
        col1, col2 = st.columns(2)
            
        with col1:
            st.image(out, caption= image_path_str + ", an image of " + Prompt)
        
        with col2:
            graphTraj(0, 0)  # 0,0 is a placeholder and will be replaced
