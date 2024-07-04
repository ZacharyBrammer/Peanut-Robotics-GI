from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

texts = ['sample text', 'example text']

dummy_pixels = torch.zeros([2, 3, 224, 224])

def embed_txt(txt):
    inputs = processor(txt, images=dummy_pixels, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs['logits_per_image']  # Logits per image is used for text
    
    return embeddings

# for i, text in enumerate(texts):
#     embedding = embed_txt(text)
#     print(f"Text: {text}")
#     print(f"Embedding shape: {embedding.shape}")
#     print(f"Embedding: {embedding.tolist()}")
#     print("=" * 50)
    