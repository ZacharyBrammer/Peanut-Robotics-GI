import lancedb
from datasets import load_dataset
from enum import Enum
import numpy
from tqdm import tqdm
import torch 
import clip 
import pyarrow as pa
# from torchvision import models, transforms
from IPython.display import display
import pandas
from PIL import Image
import os
# import cv2
import requests
from io import BytesIO

testImagePath = load_dataset("CVdatasets/ImageNet15_animals_unbalanced_aug1", split="train") #should also work with a dataset of images

class Animal(Enum):
    italian_greyhound = 0
    coyote = 1
    beagle = 2
    rottweiler = 3
    hyena = 4
    greater_swiss_mountain_dog = 5
    Triceratops = 6
    french_bulldog = 7
    red_wolf = 8
    egyption_cat = 9
    chihuahua = 10
    irish_terrier = 11
    tiger_cat = 12
    white_wolf = 13
    timber_wolf = 14


# print(testImagePath[0])
# print(Animal(testImagePath[0]["labels"]).name)

# load the pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# gen embeddings for an image
def gen_embeddings(image_path):
    image = preprocess(image_path).unsqueeze(0).to(device)
    embs = model.encode_image(image)
    return embs.detach().cpu().numpy()[0].tolist()

# embedding one image
response = requests.get("https://www.thespruce.com/thmb/iMt63n8NGCojUETr6-T8oj-5-ns=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/PAinteriors-7-cafe9c2bd6be4823b9345e591e4f367f.jpg")
img = Image.open(BytesIO(response.content))

emb_img = gen_embeddings(img)
# print("emebedded test image: ")
# print(emb_img)

# reads trajectory data file
def read_trajectory(path):
    # trajectory_data = pandas.read_csv(path)
    trajectory_data = "placeholder"
    return trajectory_data

# def store_data(db, image_path, embeddings, trajectory_data):
#     db.insert('image_embeddings', {
#         'image_path': image_path,
#         'embedding': embeddings.tolist(),
#         'trajectory_data': trajectory_data
#     })

def process_images(dir_path, trajectory_path):
    # db = lancedb.LanceDb('embeddings.db')
    db = lancedb.connect('embeddings.db')
    # db.create_table('image_embeddings', ['image_path', 'embedding', 'trajectory_data'])

    # tbl = db.create_table('image_embeddings', {
    #     'image_path': "40777060/40777060_frames/lowres_wide/40777060_98.764.png",
    #     'embedding': gen_embeddings(Image.open("40777060/40777060_frames/lowres_wide/40777060_98.764.png")),
    #     'trajectory_data': read_trajectory("40777060/40777060_frames/lowres_wide/40777060_98.764.png")
    # })

    schema = pa.schema(
        [
            pa.field("image_path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 512)),
            pa.field("trajectory_data", pa.string()),
        ]
    )

    tbl = db.create_table("image_embeddings", schema=schema, mode="overwrite")
    
    

    trajectory_data = read_trajectory(trajectory_path)
    for file in os.listdir(dir_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(dir_path, file)
            embedding = gen_embeddings(Image.open(image_path))
            # rel_data = trajectory_data.loc[trajectory_data['image']==file].to_dict(orient='records')[0]
            # store_data(db, image_path, embedding, rel_data)
            print(f"Processed and stored data for {image_path}")

            
            
            data = pandas.DataFrame({
                'image_path': image_path,
                'embedding': embedding,
                'trajectory_data': trajectory_data
                # 'trajectory_data': rel_data
            })
            tbl.add(data)

process_images("40777060/40777060_frames/lowres_wide/", "40777060/40777060_3dod_annotation.json")










# # set up lancedb table


# db = lancedb.connect("./data/tables")
# schema = pa.schema(
#     [
#         pa.field("vector", pa.list_(pa.float32(), 512)),
#         pa.field("id", pa.int32()),
#         pa.field("label", pa.int32()),
#     ]
# )
# tbl = db.create_table("images", schema=schema, mode="overwrite")

# data = []
# for i in tqdm(range(1, len(testImagePath))):
#     data.append(
#         {"vector": gen_embeddings(testImagePath[i]["img"]), "id": i, "label": testImagePath[i]["labels"]}
#     )

# tbl.add(data)
# tbl.to_pandas()


# #image search testing
# test = load_dataset("CVdatasets/ImageNet15_animals_unbalanced_aug1", split="validation")



# ##get test image
# def image_search(id):
#     print(Animal(test[id]["labels"]).name)
#     display(test[id]["img"])

#     res = tbl.search(gen_embeddings(test[id]["img"])).limit(5).to_df()
#     print(res)
#     for i in range(5):
#         print(Animal(res["label"][i]).name)
#         data_id = int(res["id"][i])
#         display(testImagePath[data_id]["img"])


# embs = gen_embeddings(test[100]["img"])

# res = tbl.search(embs).limit(1).to_df()
# print(res)

# print(Animal(res["label"][0]).name)
# id = int(res["id"][0])
# print(id)



# def image_search(id):
#     print(Animal(test[id]["labels"]).name)
#     display(test[id]["img"])

#     res = tbl.search(gen_embeddings(test[id]["img"])).limit(5).to_df()
#     print(res)
#     for i in range(5):
#         print(Animal(res["label"][i]).name)
#         data_id = int(res["id"][i])
#         display(testImagePath[data_id]["img"])

# print(image_search(1200))

# def load_image(path):
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     return image

# def embed_txt(txt):
#     text = clip.tokenize([txt]).to(device)
#     embs = model.encode_text(text)
#     return embs.detach().cpu().numpy()[0].tolist()

# print(len(embed_txt("Black and white dog")))

# def text_search(text):
#     res = tbl.search(embed_txt(text)).limit(5).to_df()
#     print(res)
#     for i in range(len(res)):
#         print(Animal(res["label"][i]).name)
#         data_id = int(res["id"][i])
#         display(testImagePath[data_id]["img"])


# text_search("a full white dog")