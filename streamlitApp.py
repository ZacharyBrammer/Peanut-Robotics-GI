import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Lorem ipsum dolor!

"""

with st.form("prompt"):
   st.write("Prompt goes here:")
   Prompt=st.text_input("Prompt")
   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write(Prompt)
