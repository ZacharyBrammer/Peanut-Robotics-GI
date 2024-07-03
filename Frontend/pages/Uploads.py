import streamlit as st
import os

st.set_page_config(page_title="Uploads")
# Ensure the 'uploads' directory exists
if not os.path.exists('Frontend/uploads'):
    os.makedirs('Frontend/uploads')

st.title('File Upload Example')

uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join('Frontend/uploads', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved file: {uploaded_file.name}")
        st.image('Frontend/uploads/'+uploaded_file.name, caption='lorem ipsum')