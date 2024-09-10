import cv2
import torch
from yolov5 import YOLOv5
import torch


# Убедитесь, что путь к файлу правильный
pth_model_path = 'X:/PROJECTS/TEST_FOLDER/TEST_1_INITALIZATING_CAMERA/coco/your_model.pth'

# Загрузите модель
try:
    model = torch.load(pth_model_path, map_location='cpu')
except FileNotFoundError:
    print(f"Файл модели не найден: {pth_model_path}")


# Путь к вашему файлу модели .pth
pth_model_path = 'X:/PROJECTS/TEST_FOLDER/TEST_1_INITALIZATING_CAMERA/coco/your_model.pth'

# Загрузите модель из файла .pth
model = torch.load(pth_model_path, map_location='cpu')

# Сохраните модель в формате .pt
pt_model_path = 'X:/PROJECTS/TEST_FOLDER/TEST_1_INITALIZATING_CAMERA/coco/your_model.pt'
torch.save(model, pt_model_path)

# Путь к вашему файлу модели .pth
model_path = 'X:/PROJECTS/TEST_FOLDER/TEST_1_INITALIZATING_CAMERA/coco/your_model.pth'


# Путь к вашему файлу модели .pt
model_path = 'X:/PROJECTS/TEST_FOLDER/TEST_1_INITALIZATING_CAMERA/coco/your_model.pt'

# Инициализация модели YOLOv5
model = YOLOv5(model_path, device='cpu')

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    # Захват кадра из камеры
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр")
        break

    # Преобразование изображения в формат, необходимый модели
    results = model.predict(frame)

    # Отображение результатов на кадре
    for detection in results.pandas().xyxy[0].itertuples():
        x1, y1, x2, y2, conf, cls = detection[1:7]
        if conf > 0.5:  # Порог уверенности
            label = f"Object {cls} ({conf:.2f})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображение изображения
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
