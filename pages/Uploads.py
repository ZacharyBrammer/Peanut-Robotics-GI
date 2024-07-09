import streamlit as st
import os
import zipfile
from Utils import ImageConvert


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

uploaded_files = st.file_uploader("Choose image files", type=["zip"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join('uploads', uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        databasename= os.path.splitext(uploaded_file.name)[0]
        st.success(f"Saved file: {uploaded_file.name}")
        with zipfile.ZipFile(uploaded_file, "r") as z:
            z.extractall('uploads')
            
        table = ImageConvert.process_images(os.path.join('./uploads', databasename, databasename+'_frames', 'lowres_wide'), os.path.join('./uploads', databasename, databasename+'_frames', 'lowres_wide.traj'))

