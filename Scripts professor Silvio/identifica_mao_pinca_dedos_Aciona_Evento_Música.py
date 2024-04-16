import cv2
import mediapipe as mp
import pygame

class Retangulo:
    def __init__(self, pos_x, pos_y, largura, altura):
        self.retangulo_x = pos_x
        self.retangulo_y = pos_y
        self.retangulo_largura = largura
        self.retangulo_altura = altura

    def verifica_pinca_no_retangulo(self, pos_val_x, pos_val_y):
        return self.retangulo_x <= pos_val_x <= self.retangulo_x + self.retangulo_largura and \
               self.retangulo_y <= pos_val_y <= self.retangulo_y + self.retangulo_altura

class DetectaMao:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.retangulo_1 = Retangulo(50, 50, 80, 80)
        self.retangulo_2 = Retangulo(50, 150, 80, 80)
        pygame.mixer.init()

    def verificar_movimento_pinca(self, landmarks, image):
        dedo_indicador = landmarks[8]
        dedo_polegar = landmarks[4]

        dist_X = dedo_indicador[0] - dedo_polegar[0]
        dist_Y = dedo_indicador[1] - dedo_polegar[1]

        pinca_no_retangulo_1 = self.retangulo_1.verifica_pinca_no_retangulo(dedo_indicador[0], dedo_indicador[1])
        pinca_no_retangulo_2 = self.retangulo_2.verifica_pinca_no_retangulo(dedo_indicador[0], dedo_indicador[1])

        if -20 <= dist_X <= 20 and -20 <= dist_Y <= 20:
            if pinca_no_retangulo_1:
                pygame.mixer.music.load('mrroboto.mp3')
                pygame.mixer.music.play(-1)
            elif pinca_no_retangulo_2:
                pygame.mixer.music.stop()

    def run(self):
        with self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        landmarks = []
                        for point in hand_landmarks.landmark:
                            x = int(point.x * image.shape[1])
                            y = int(point.y * image.shape[0])
                            landmarks.append((x, y))

                        self.verificar_movimento_pinca(landmarks, image)

                cv2.rectangle(image, (self.retangulo_1.retangulo_x, self.retangulo_1.retangulo_y),
                              (self.retangulo_1.retangulo_x + self.retangulo_1.retangulo_largura,
                               self.retangulo_1.retangulo_y + self.retangulo_1.retangulo_altura), (100, 80, 255), 2)

                cv2.rectangle(image, (self.retangulo_2.retangulo_x, self.retangulo_2.retangulo_y),
                              (self.retangulo_2.retangulo_x + self.retangulo_2.retangulo_largura,
                               self.retangulo_2.retangulo_y + self.retangulo_2.retangulo_altura), (100, 80, 255), 2)

                cv2.imshow('Detecção de Mão', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        pygame.mixer.quit()
        self.cap.release()
        cv2.destroyAllWindows()

# Executar o código
detecta_mao = DetectaMao()
detecta_mao.run()



