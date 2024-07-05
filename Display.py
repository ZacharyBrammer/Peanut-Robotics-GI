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
col1, col2 = st.columns(2)

# Create a form
with col1:
    with st.form("prompt"):
        st.write("Prompt goes here:")
        Prompt = st.text_input("Prompt")
        # Every form must have a submit button
        submitted = st.form_submit_button("Submit")
    
        if submitted:
           st.write(Prompt)
           embedPrompt = embed_txt(Prompt)
           out = "static/1.jpg" #placeholder. output goes here
           # Use columns to place the image and graph side by side

           st.image(out, caption= out + ", an image of " + Prompt)
        

        
with col2:
        if submitted:
            graphTraj(0, 0)  # 0,0 is a placeholder and will be replaced
