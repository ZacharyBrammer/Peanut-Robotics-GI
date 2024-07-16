import os
from io import BytesIO

from enum import Enum
from tqdm import tqdm
import requests
from pathlib import Path
import regex as re

import torch
import clip
import lancedb

import pyarrow as pa
import pandas
from PIL import Image

from Utils import rotateImage


# load the pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
# original model ViT-B/32

def gen_embeddings(image_path):
    image = preprocess(image_path).unsqueeze(0).to(device)
    embs = model.encode_image(image)
    return embs.detach().cpu().numpy()[0].tolist()

# reads trajectory data file


def read_trajectory(path):
    trajectory_data = pandas.read_csv(path)
    return trajectory_data


def extract_number_from_filename(filename):
    pattern = r'\d+\.\d+\.png$'

    # Use re.search to find the pattern in the filename
    match = re.search(pattern, filename)

    if match:
        number_str = match.group(0)

        number = float(number_str[:-4])

        return number
    else:
        return -1


def process_images(dir_path, trajectory_path):
    db = lancedb.connect('embeddings.db')

    schema = pa.schema(
        [
            pa.field("image_path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 1024)),
            pa.field("trajectory_data", pa.string()),
        ]
    )

    tbl = db.create_table(
        str(dir_path.split('/')[2]), schema=schema, mode="overwrite")
    print("Created Table for " + str(Path(dir_path.split('/')[2])))

    data = []

    counter = 0
    trajectories = open(trajectory_path, "r").read().split("\n")

    files = os.listdir(dir_path)
    files.sort(key=extract_number_from_filename)

    # image size: 256 x 192
    # Leftmost, Uppermost, Rightmost, Bottom

    for file in tqdm(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):

            image_path = os.path.join(dir_path, file)
            rotatedImages = rotateImage.rotate_image(Image.open(image_path))

            boxes = [(0, 0, rotatedImages.width/2, rotatedImages.height/2), (rotatedImages.width/2, 0, rotatedImages.width, rotatedImages.height/2), (0, rotatedImages.height /
                                                                                                                                                      2, rotatedImages.width/2, rotatedImages.height), (rotatedImages.width/2, rotatedImages.height/2, rotatedImages.width, rotatedImages.height)]

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
