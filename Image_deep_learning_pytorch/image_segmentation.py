import torch, torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# semantic vs instance segmentation

image = Image.open("images/British_shorthair_36.jpg")
mask = Image.open("annots/British_shorthair_36.jpg")

transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image)
mask_tensor = transform(mask)

binary_mask = torch.where(
    mask_tensor == 1/255,
    torch.tensor(1.0),
    torch.tensor(0.0)
)

to_pil_image = transforms.ToPILImage()
mask = to_pil_image(binary_mask)
plt.imshow(mask)

object_tensor = image_tensor * binary_mask
to_pil_image = transforms.ToPILImage()
object_image = to_pil_image(object_tensor)

plt.imshow(object_image)