import os

import streamlit as st
from PIL import Image

from Utils.ImageSearch import imageSearch
from Utils.TextConvert import embed_txt
from Utils.trajectoryGraph import graphTraj

st.set_page_config(
    page_title="Find and Display"
)

st.title("Peanut Robotics Spatial Image Search")

folder_path = "./embeddings.db"
file_list = os.listdir(folder_path)

if not file_list:
    file_list = ['No embeddings found!']

selected_ds = st.selectbox('Select a table', file_list).split('.')[0]

with st.form("prompt"):
    st.write("Prompt goes here:")
    Prompt = st.text_input("Prompt")

    submitted = st.form_submit_button("Submit")

    if submitted:
        embbededText = embed_txt(Prompt)
        imag = imageSearch(embbededText, selected_ds)
        if imag.empty:
            st.write(
                "No data found, please try a different dataset or reupload the current one.")
        else:
            image_path_str = str(imag.image_path.iloc[0])
            out = Image.open(image_path_str)

            col1, col2 = st.columns(2)

            trajectory = imag['trajectory_data'].array[0].split()

            imag_x = float(trajectory[-3])
            imag_y = float(trajectory[-2])

            with col1:
                st.image(out, width=330, caption=image_path_str +
                         ", an image of " + Prompt)

            with col2:

                trajpath = "./uploads/"+selected_ds+"/"+selected_ds+"_frames/lowres_wide.traj"
                graphTraj(imag_x, imag_y, trajpath,
                          image_path_str.split("/")[5])
