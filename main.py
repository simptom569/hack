import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder

data_path = 'datasets/train'
output_path = 'datasets/frames'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Преобразование видео в кадры
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    output_category_path = os.path.join(output_path, category)

    if not os.path.exists(output_category_path):
        os.makedirs(output_category_path)

    for video in tqdm(os.listdir(category_path)):
        video_path = os.path.join(category_path, video)
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0

        while success:
            frame_path = os.path.join(output_category_path, f"frame{count}.jpg")
            cv2.imwrite(frame_path, image)
            success, image = cap.read()
            count += 1

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_classes = 24
batch_size = 20
learning_rate = 0.08
num_epochs = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = ImageFolder(root='datasets/frames', transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ActionRecognitionModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in tqdm(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Testing Accuracy: {accuracy}')

torch.save(model.state_dict(), 'action_recognition_model.pth')
