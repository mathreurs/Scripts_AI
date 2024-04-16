import cv2
import os

# MELHORAS PARA O CÓDIGO
# Controlar o momento da captura da imagem
# Controlar a qualidade da imagem
# Acertar o lugar da imagem(assunto)
# Controlar o tamanho da área de captura

# Carrega o classificador de faces pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Verifica se o diretório de imagens já existe, caso contrário, cria-o
if not os.path.exists('imagens'):
    os.makedirs('imagens')

# Dicionário para controlar o número de imagens capturadas por pessoa
captured_images = {}

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

        # Captura a face e salva como uma imagem jpg
        face_image = frame[y:y+h, x:x+w]

        # Verifica se a pessoa já foi capturada duas vezes
        if captured_images.get(tuple(face_image.shape[:2]), 0) < 2:
            # Incrementa o contador de imagens para a pessoa
            captured_images[tuple(face_image.shape[:2])] = captured_images.get(tuple(face_image.shape[:2]), 0) + 1

            # Salva a imagem capturada
            image_counter = captured_images[tuple(face_image.shape[:2])]
            cv2.imwrite(f'imagens/face{image_counter:02d}.jpg', face_image)

    # Exibe o frame resultante
    cv2.imshow('Face Detection', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Verifica se todas as pessoas tiveram duas imagens capturadas
    if all(value == 2 for value in captured_images.values()):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
