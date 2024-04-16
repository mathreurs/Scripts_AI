import cv2
import mediapipe as mp

# Inicializar o MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Definir as dimensões da janela
largura_tela = 640
altura_tela = 480

# Configurar desenho de anéis amarelos
desenho_aneis = mp.solutions.drawing_utils

def detectar_mao():
    # Inicializar o objeto de captura de vídeo
    captura = cv2.VideoCapture(0)

    while True:
        # Ler o quadro atual do vídeo
        ret, quadro = captura.read()

        # Inverter o quadro horizontalmente
        quadro = cv2.flip(quadro, 1)

        # Converter o quadro para RGB para o MediaPipe
        quadro_rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)

        # Detectar as mãos no quadro
        resultado = hands.process(quadro_rgb)

        if resultado.multi_hand_landmarks:
            for mao_landmarks in resultado.multi_hand_landmarks:
                for id, ponto in enumerate(mao_landmarks.landmark):
                    # Obter as coordenadas normalizadas do ponto
                    posicao_x = int(ponto.x * largura_tela)
                    posicao_y = int(ponto.y * altura_tela)

                    # Verificar se é o dedo indicador (ID = 8)
                    if id == 8:
                        # Desenhar um círculo nas pontas dos dedos indicadores
                        cv2.circle(quadro, (posicao_x, posicao_y), 10, (0, 255, 255), -1)

                        # Exibir as coordenadas do dedo indicador
                        cv2.putText(quadro, f'({posicao_x}, {posicao_y})', (posicao_x, posicao_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            # Desenhar a detecção de mãos no quadro
            desenho_aneis.draw_landmarks(quadro, mao_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar o quadro capturado
        cv2.imshow("Detecção de Mão", quadro)

        # Verificar se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    captura.release()
    cv2.destroyAllWindows()

# Chamar a função principal para executar a detecção de mão
detectar_mao()