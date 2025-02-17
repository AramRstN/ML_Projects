import torch
import clip
from PIL import Image
import requests
from io import BytesIO


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device = device)

#Pretrained
# Computing the similarity between image and text

def text_image_match(image_url, text):

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    the_image = preprocess(image).unsqueeze(0).to(device)

    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        img_features = model.encode_image(the_image)
        text_features = model.encode_text(text)

    img_features /= img_features.norm(dim=1, keepdim= True)
    text_features /= text_features.norm(dim=1, keepdim= True)

    img_txt_similarity = (100.0 * img_features @ text_features.T).softmax(dim= -1)

    return img_txt_similarity


image_url = input("Enter an image URL:")
text = input("Enter a description:").split(',')

image_text_similarity = text_image_match(image_url, text)

for i, caption in enumerate(text):
    print(f"Caption: {caption.strip()} --> Similarity: {image_text_similarity[0,i].item():.4f}")

