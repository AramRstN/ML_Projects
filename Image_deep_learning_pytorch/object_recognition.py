import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor()] #useful for bounding box
)

image = Image.open('cat01.jpg')
image_tensor = transforms(image)
image_reshaped = image_tensor.unsqueeze(0)

image_tensor = transform(image)

box = [10, 10, 200, 200] #X_min/max, y_min/max
bbox_tensor = torch.tensor(box)
bbox_tensor = bbox_tensor.unsqueeze(0)

bbox_image = draw_bounding_boxes(
    image_tensor, bbox_tensor, width=3, colors='red'
)

transform_bbx = transforms.Compose([
    transforms.ToPILImage() 
])
pil_image = transform_bbx(bbox_image)

plt.imshow(pil_image)