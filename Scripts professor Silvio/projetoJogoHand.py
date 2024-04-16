import cv2
import mediapipe
from cvzone.HandTrackingModule import HandDetector
import socket

#parametros
width, height = 1000,600

# formato da webcam
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#detector as mãos
detector = HandDetector(maxHands = 2, detectionCon =0.8 )

#comunicação
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("192.168.0.180", 2804)

while True:
    #captura o frame da webcam
    success, img = cap.read()
    #detecta de mãos
    hands, img = detector.findHands(img)
    
    data = []
    #valores das landmarks (x,y,z) * 21
    if hands:
        #pega o valor da1 primeira mão
        hand = hands[0]
        #pega a lista de coordenadas das landmarks
        lmList = hand['lmList']
        # print(lmList)
        for lm in lmList:
            data.extend([lm[0],height - lm[1],lm[2]])
        # print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
    
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
