import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import cv2
from PIL import Image
import os

# Задайте путь к обученной модели
model_path = 'action_recognition_model.pth'

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Задайте функции для предобработки изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Загрузите обученную модель
num_classes = 24
model = ActionRecognitionModel(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Задайте путь к видео
video_path = '5 men clapping.mp4'

# Загрузите видео и прочитайте кадры
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_num in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # Сохраните кадр как изображение
    frame_path = f'temp_frame_{frame_num}.jpg'
    cv2.imwrite(frame_path, frame)

    # Предобработайте изображение и примените модель
    image_tensor = preprocess_image(frame_path)
    with torch.no_grad():
        outputs = model(image_tensor)

    # Получите предсказанный класс
    _, predicted_class = torch.max(outputs, 1)
    predicted_class = predicted_class.item()

    # Выведите предсказание
    print(f'Frame {frame_num}: Predicted Action - Class {predicted_class}')
    cv2.imshow('frame', frame)
    cv2.waitKey(5)

    # Удалите временное изображение
    os.remove(frame_path)

# Закройте видеопоток
cap.release()
cv2.destroyAllWindows()