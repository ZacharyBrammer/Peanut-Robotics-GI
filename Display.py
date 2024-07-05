import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import sys

from Utils.TextConvert import embed_txt

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
        embedPrompt = embed_txt(Prompt)
        st.write(f"Text: {Prompt}")
        st.write(f"Embedding shape: {embedPrompt.shape}")
        st.write(f"Embedding: {embedPrompt.tolist()}")
        st.write("=" * 50)
        st.image('static/1.jpg', caption='placeholder1')
        st.image('static/2.jpg', caption='placeholder2')
        st.image('static/3.jpg', caption='placeholder3')