import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
# import random

from model import CNN
import inspect

print(CNN)
print(inspect.getsource(CNN))

import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device used:{device}")

transform=transforms.ToTensor()
train_dataset=datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)
# print(train_dataset.classes)
# print(len(train_dataset))
# print(train_dataset[0])
image, label = train_dataset[0]

# print(image.shape)
# print(label)
# plt.imshow(image.permute(1, 2, 0))
# plt.title(train_dataset.classes[label])
# plt.show()

# for i in range(5):
#     index = random.randint(0, len(train_dataset) - 1)

#     image, label = train_dataset[index]

#     plt.figure(figsize=(3, 3))
#     plt.imshow(image.permute(1, 2, 0))
#     plt.title(train_dataset.classes[label])
#     plt.axis("off")
#     plt.show()

train_loader=DataLoader(
    train_dataset,
    batch_size=25,
    shuffle=True
)

valid_dataset = datasets.ImageFolder(
    root="dataset/valid",
    transform=transform
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=25,
    shuffle=False
)

images, labels = next(iter(train_loader))

# print(images.shape)
# print(labels.shape)

model = CNN()
model = model.to(device)

loss_func=nn.CrossEntropyLoss()

optimiser=optim.Adam(model.parameters(),lr=0.001)

epochs=10
best_valid_accuracy = 0
for epoch in range(epochs):
    model.train()

    correct=0
    total=0
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)

        optimiser.zero_grad()

        output=model(images)

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loss=loss_func(output,labels)

        loss.backward()

        optimiser.step()

    accuracy=100*correct/total

    print(f'Epoch: {epoch+1}/{epochs},Loss:{loss.item():.4f},Accuracy:{accuracy:.2f}%')

    model.eval()
    with torch.no_grad():
        valid_correct=0
        valid_total=0
        for images,labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            output=model(images)

            _,predicted=torch.max(output,1)

            valid_correct+=(predicted==labels).sum().item()
            valid_total+=labels.size(0)
        
        valid_accuracy=100*valid_correct/valid_total

        print(f'Validation Accuracy:{valid_accuracy:.2f}%')

        if valid_accuracy > best_valid_accuracy:
            best_val_accuracy = valid_accuracy
            torch.save(model.state_dict(), "rvf_model.pth")
            print("RealVsFake model saved!")

# torch.save(model.state_dict(), "real_vs_fake_model.pth")
# print("Model saved successfully!")