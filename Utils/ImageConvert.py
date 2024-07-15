import lancedb
from enum import Enum
from tqdm import tqdm
import torch 
import clip
import pyarrow as pa
import pandas
from PIL import Image
import os
import requests
from io import BytesIO
import regex as re
from Utils import rotateImage
from PIL import Image
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

    
    tbl = db.create_table(str(dir_path.split('/')[2]), schema=schema, mode="overwrite")
    print("Created Table for " + str(Path(dir_path.split('/')[2])))

    data = []

    counter = 0
    trajectories = open(trajectory_path, "r").read().split("\n")

    files = os.listdir(dir_path)
    files.sort(key=extract_number_from_filename)

    # image size: 256 x 192
    # Leftmost, Uppermost, Rightmost, Bottom        

    for file in tqdm(files):
        print(file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            
            image_path = os.path.join(dir_path, file)
            rotatedImages = rotateImage.rotate_image(Image.open(image_path))
            # embedding = gen_embeddings(rotatedImages)
            
            boxes = [(0, 0, rotatedImages.width/2, rotatedImages.height/2), (rotatedImages.width/2, 0, rotatedImages.width, rotatedImages.height/2), (0, rotatedImages.height/2, rotatedImages.width/2, rotatedImages.height), (rotatedImages.width/2, rotatedImages.height/2, rotatedImages.width, rotatedImages.height)]

            for box in boxes:
                embedding = gen_embeddings(rotatedImages.crop(box))

                data.append({
                    'image_path': image_path,
                    'embedding': embedding,
                    'trajectory_data': trajectories[counter]
                })

            counter += 1

    tbl.add(data)

    return tbl

# process_images("40777060/40777060_frames/lowres_wide", "40777060/40777060_frames/lowres_wide.traj")
