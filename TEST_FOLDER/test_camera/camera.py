import cv2
import mediapipe as mp

# Инициализация Mediapipe для рук и лиц
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Открытие видеопотока с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Преобразование изображения в формат RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка изображения для обнаружения рук и лиц
    result_hands = hands.process(rgb_frame)
    result_face = face_detection.process(rgb_frame)

    # Отрисовка обнаруженных лиц
    if result_face.detections:
        for detection in result_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отрисовка обнаруженных рук
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Показ изображения
    cv2.imshow('Hand and Face Detection', frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
