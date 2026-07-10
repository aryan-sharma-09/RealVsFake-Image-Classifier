import torch 
from torchvision import transforms
from PIL import Image

from model import CNN

model=CNN()

weights=torch.load("rvf_model.pth")
model.load_state_dict(weights)

model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image=Image.open("")

image=transform(image)
image=image.unsqueeze(0)

with torch.no_grad():
    output=model(image)
    _,predicted=torch.max(output,1)

if predicted.item()==0:
    print("Prediction : Fake Image")
else:
    print("Prediction : Real Image")