import cv2
import numpy as np

# Carrega a imagem da câmera
cap = cv2.VideoCapture(0)

while True:
    # Captura uma imagem da câmera
    ret, frame = cap.read()

    # Converte a imagem para cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta pontos de interesse no corpo humano
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray_frame, None)

    # Encontra os pontos de interesse mais significativos
    thresh = 0.01
    min_distance = float("inf")
    good_corners = []
    for i in range(len(kp)):
        with np.errstate(divide='ignore', invalid='ignore'):
            score = des[i].dot(des.T) / (kp[i].response * kp[i].response)
            if score > thresh and des[i].var() > min_distance:
                good_corners.append(kp[i].pt)
                min_distance = des[i].var()

    # Desenha os pontos de interesse na imagem original
    for i in range(len(good_corners)):
        cv2.circle(frame, (int(good_corners[i][0]), int(good_corners[i][1])), 5, (0, 0, 255), -1)

    # Exibe a imagem resultante
    cv2.imshow("Frame", frame)

    # Verifica se o usuário pressionou 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()