import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
import random


batch_size = 8
learning_rate = 0.01
num_epochs = 10
sequence_length = 24


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, image = cap.read()
    while success:
        frames.append(image)
        success, image = cap.read()
    return frames


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frames, label = self.samples[index]
        if self.transform:
            frames = self.transform(frames)
        return frames, label

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir):
        frames_dataset = []
        for cls_index, target_class in enumerate(self.classes):
            class_dir = os.path.join(dir, target_class)
            for video_filename in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_filename)
                frames = self._read_video_frames(video_path)
                frames_dataset.append((frames, cls_index))
        return frames_dataset

    def _read_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
        return frames


class VideoSequenceResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        resized_frames = [cv2.resize(frame, self.size) for frame in frames]
        return resized_frames

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, sequence_length=sequence_length):
        super(ActionRecognitionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, seq_len, -1)
        return x.mean(dim=1)

if __name__ == "__main__":
    data_path = 'datasets/train'
    output_path = 'datasets/frames'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        output_category_path = os.path.join(output_path, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for video in tqdm(os.listdir(category_path)):
            video_path = os.path.join(category_path, video)
            frames = read_video_frames(video_path)
            frame_count = len(frames)

            for count, frame in enumerate(frames):
                frame_path = os.path.join(output_category_path, f"{video}_frame{count}.jpg")
                cv2.imwrite(frame_path, frame)
   
    transform = transforms.Compose([
    VideoSequenceResize((224, 224)),
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),
    transforms.Lambda(lambda frames: torch.stack(frames)),
    ])

    dataset = VideoFolderDataset(root='datasets/frames', transform=transform)

    num_classes = len(dataset.classes)

    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ActionRecognitionModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        
        accuracy_train = correct_train / total_train

        print(f'Эпоха {epoch + 1}/{num_epochs}, ',
              f'Тренировочная потеря: {total_loss / len(train_loader)}, '
              f'Тренировочная точность: {accuracy_train}')

        model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0

            for inputs_val, labels_val in tqdm(val_loader):
                outputs_val = model(inputs_val)
                softmax_outputs_val = torch.softmax(outputs_val, dim=1)
                _, predicted_val = torch.max(softmax_outputs_val, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

            accuracy_val = correct_val / total_val
            print(f'Эпоха {epoch + 1}/{num_epochs}, Валидационная точность: {accuracy_val}')

            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))

    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0

        for inputs_test, labels_test in tqdm(test_loader):
            outputs_test = model(inputs_test)
            softmax_outputs_test = torch.softmax(outputs_test, dim=1)
            _, predicted_test = torch.max(softmax_outputs_test, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()

        accuracy_test = correct_test / total_test
        print(f'Тестовая точность: {accuracy_test}')

    torch.save(model.state_dict(), 'action_recognition_model.pth')
