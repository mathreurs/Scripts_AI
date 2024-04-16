import cv2

#ver em opencv/data/haarcascades

# Carrega o classificador pré-treinado para detecção de corpo inteiro
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicializa a câmera
cap = cv2.VideoCapture(0)

while True:
    # Captura o frame da câmera
    ret, frame = cap.read()

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta os corpos no frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha um retângulo ao redor dos corpos detectados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Exibe o frame resultante
    cv2.imshow('Body Detection', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()





