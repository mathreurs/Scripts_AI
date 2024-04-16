import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa o MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Inicializa a câmera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Captura o frame da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para RGB para o MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta mãos no frame
    results_hands = hands.process(frame_rgb)

    # Detecta rostos no frame
    results_face = face_detection.process(frame_rgb)

    # Desenha retângulos ao redor das mãos detectadas
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Desenha retângulos ao redor dos rostos detectados
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    # Exibe o frame resultante
    cv2.imshow('Face and Hand Tracking', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
