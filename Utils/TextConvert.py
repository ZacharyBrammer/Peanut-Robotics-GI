import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

def embed_txt(txt):
    text = clip.tokenize([txt]).to(device)
    embs = model.encode_text(text)
    return embs.detach().cpu().numpy()[0].tolist()

print(clip.available_models())