import cv2

def get_user_direction():
    # Carrega o classificador de cascata dos olhos
    eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Inicializa a captura da câmera
    cap = cv2.VideoCapture(0)

    while True:
        # Captura uma imagem do usuário
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar a imagem. Tente ajustar a iluminação ou posição da câmera.")
            continue

        # Converte a imagem para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Encontra os contornos nos olhos do usuário
        eyes = eyes_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Se pelo menos dois olhos forem detectados, determina a direção em que o usuário está olhando
        if len(eyes) == 2:
            break

    while True:
        # Captura uma imagem do usuário
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar a imagem. Tente ajustar a iluminação ou posição da câmera.")
            continue

        # Converte a imagem para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Encontra os contornos nos olhos do usuário
        eyes = eyes_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Determina a direção em que o usuário está olhando
        if len(eyes) == 2:

            eye_x, eye_y, eye_w, eye_h = eyes[0]
            left_eye_center_x = eye_x + eye_w / 2
            eye_x, eye_y, eye_w, eye_h = eyes[1]
            right_eye_center_x = eye_x + eye_w / 2

            user_direction = (right_eye_center_x - left_eye_center_x) / (right_eye_center_x + left_eye_center_x) * 100

            print(f"Direção do olhar: {user_direction} ")

            if int(user_direction) > -8 and int(user_direction) < 8:
                print("O usuário está olhando em direção ao centro.")
            elif user_direction > 0:
                print("O usuário está olhando para a direita.")
            else:
                print("O usuário está olhando para a esquerda.")

        # Exibe a imagem com os olhos detectados
        cv2.imshow('User Direction Detector', frame)

        # Verifica se o usuário pressionou a tecla 'q' para sair do programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a captura da câmera e fecha a janela de visualização
    cap.release()
    cv2.destroyAllWindows()

# Chama a função para determinar a direção do olhar
get_user_direction()