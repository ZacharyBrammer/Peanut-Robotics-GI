import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_txt(txt):
    text = clip.tokenize([txt]).to(device)
    embs = model.encode_text(text)
    return embs.detach().cpu().numpy()[0].tolist()

# print(embed_txt("Black and white dog"))

    # def embed_txt(txt):
#     inputs = processor(txt, images=dummy_pixels, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs['logits_per_image']  # Logits per image is used for text

    
    
#     return embeddings