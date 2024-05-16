import os 
import cv2
import time
import uuid

IMAGE_PATH = 'collectedImages'

labels = ['oi','Sim','Nâo','Obrigado','Eu te amo','Por favor']

number_of_images = 20

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path)
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(5)
    for imgNum in range(number_of_images):
        ret, frame = cap.read()
        imageName = os.path.join(IMAGE_PATH,label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imageName, frame)
        cv2.imshow('frame',frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()