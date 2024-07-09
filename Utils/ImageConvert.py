import lancedb
from datasets import load_dataset
from enum import Enum
import numpy
from tqdm import tqdm
import torch 
import clip
import pyarrow as pa
from IPython.display import display
import pandas
from PIL import Image
import os
import requests
from io import BytesIO
import trajectory
import regex as re
from Utils import rotateImage
from PIL import Image
import clip
import sys
from pathlib import Path



# load the pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
#original model ViT-B/32

# gen embeddings for an image
def gen_embeddings(image_path):
    image = preprocess(image_path).unsqueeze(0).to(device)
    embs = model.encode_image(image)
    return embs.detach().cpu().numpy()[0].tolist()

# embedding one image
response = requests.get("https://www.thespruce.com/thmb/iMt63n8NGCojUETr6-T8oj-5-ns=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/PAinteriors-7-cafe9c2bd6be4823b9345e591e4f367f.jpg")
img = Image.open(BytesIO(response.content))

emb_img = gen_embeddings(img)

# reads trajectory data file
def read_trajectory(path):
    trajectory_data = pandas.read_csv(path)
    return trajectory_data


def extract_number_from_filename(filename):
    pattern = r'\d+\.\d+\.png$'
  
    # Use re.search to find the pattern in the filename
    match = re.search(pattern, filename)
  
    if match:
        # Extract the matched part (number at the end)
        number_str = match.group(0)
      
        # Convert to float if needed
        number = float(number_str[:-4])  # Convert to float and remove ".png"
      
        return number
    else:
        return -1  # Return -1 if no match found


def process_images(dir_path, trajectory_path):
    db = lancedb.connect('embeddings.db')

    schema = pa.schema(
        [
            pa.field("image_path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 1024)),
            pa.field("trajectory_data", pa.string()),
        ]
    )

    # tbl = db.create_table(str(Path(dir_path).parent.parent), schema=schema, mode="overwrite")
    # print("Created Table for " + str(Path(dir_path).parent.parent))
    tbl = db.create_table(str(dir_path.split('/')[1]), schema=schema, mode="overwrite")
    print("Created Table for " + str(Path(dir_path.split('/')[1])))

    data = []

    counter = 0
    trajectories = open(trajectory_path, "r").read().split("\n")
    files = os.listdir(dir_path)
    files.sort(key=extract_number_from_filename)

    for file in tqdm(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(dir_path, file)
            rotatedImages = rotateImage.rotate_image(Image.open(image_path))
            embedding = gen_embeddings(rotatedImages)
           
            data.append({
                'image_path': image_path,
                'embedding': embedding,
                'trajectory_data': trajectories[counter]
            })

            counter += 1

    tbl.add(data)

    return tbl


# path1 = sys.argv[1] #Images
# path2 = sys.argv[2] #Trajectory
# table = process_images(path1, path2)

#40777060/40777060_frames/lowres_wide/
#40777060/40777060_frames/lowres_wide.traj
#res = table.search(gen_embeddings(Image.open("40777060/40777060_frames/lowres_wide/40777060_98.764.png"))).limit(5).to_pandas()
