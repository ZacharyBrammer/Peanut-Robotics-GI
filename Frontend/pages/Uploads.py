import streamlit as st
import os

# Define the path to the uploads directory in the root folder
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')

# Ensure the 'uploads' directory exists
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

st.set_page_config(page_title="Uploads")
# Ensure the 'uploads' directory exists
# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

st.title('Image Upload')

uploaded_files = st.file_uploader("Choose image files", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join('uploads', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved file: {uploaded_file.name}")
        st.image('uploads/'+uploaded_file.name, caption='lorem ipsum')