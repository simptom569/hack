import cv2
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from learn import ActionRecognitionModel, VideoSequenceResize
    
# Функция для загрузки видео и разделения его на последовательности кадров
def read_video_sequence(video_path, sequence_length=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, image = cap.read()
    while success:
        frames.append(image)
        success, image = cap.read()

    # Разбиваем видео на последовательности
    sequences = [frames[i:i + sequence_length] for i in range(0, len(frames), sequence_length)]

    return sequences

# Преобразование последовательности кадров с помощью тех же преобразований, что и при обучении
transform = transforms.Compose([
    VideoSequenceResize((224, 224)),
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),
])

# Загружаем модель
model = ActionRecognitionModel(num_classes=24)
model.load_state_dict(torch.load('action_recognition_model.pth'))
model.eval()

# Путь к видео для распознавания
video_path = 'gg.mp4'

# Читаем видео и создаем последовательности кадров
video_sequences = read_video_sequence(video_path, sequence_length=10)

# Проходим по каждой последовательности и делаем предсказания
for i, sequence in enumerate(video_sequences):
    # Применяем преобразования
    input_sequence = transform(sequence)

    # Добавляем размерность батча
    input_sequence = input_sequence.unsqueeze(0)

    # Передаем через модель
    with torch.no_grad():
        output = model(input_sequence)

    # Извлекаем предсказанный класс для каждой последовательности
    predicted_classes = torch.argmax(output, dim=1).numpy()

    # Check Data Types
    print(predicted_classes.dtype)

    # Check Array Elements
    print(predicted_classes)

    # Convert to integers if needed
    predicted_classes = predicted_classes.astype(int)

    # Handle Unexpected Data, if necessary

    overall_action_class = np.bincount(predicted_classes.flatten()).argmax()
    print(f"Sequence {i + 1}: Overall Action {overall_action_class}")