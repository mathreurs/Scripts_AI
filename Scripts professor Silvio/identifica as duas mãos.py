import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

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

    # Desenha retângulos ao redor das mãos detectadas
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Exibe o frame resultante
    cv2.imshow('Hand Tracking', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
