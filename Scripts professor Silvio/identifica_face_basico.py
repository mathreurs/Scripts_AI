import cv2

# Carrega o classificador de faces pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a câmera
cap = cv2.VideoCapture(0)

while True:
    # Captura o frame da câmera
    ret, frame = cap.read()

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta as faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha um retângulo ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Exibe o frame resultante
    cv2.imshow('Face Detection', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
