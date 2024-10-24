import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
from threading import Thread
from queue import Queue
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Get available voices and set to Portuguese voice
voices = engine.getProperty('voices')
portuguese_voice = None
for voice in voices:
    # Look for Portuguese or Brazilian voice
    if "portuguese" in voice.name.lower() or "brazil" in voice.name.lower():
        portuguese_voice = voice
        break

# Set voice properties
if portuguese_voice:
    engine.setProperty('voice', portuguese_voice.id)
else:
    print("Aviso: Voz em português não encontrada. Usando voz padrão.")

# Optimize voice settings for Portuguese
engine.setProperty('rate', 110)      # Ajuste de velocidade de fala
engine.setProperty('volume', 1)      # Volume
engine.setProperty('pitch', 100)      # Ajuste para tornar fala mais clara

# Cria uma fila para dizer o que foi detectado
speech_queue = Queue()
last_spoken_time = time.time()
last_spoken_text = ""

# Flag to control the speech thread
speech_thread_running = True

def speak_worker():
    """
    Função que é executada em uma parte separada para converter texto em fala
    """
    global speech_thread_running
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=1)
            # Add slight pause between words for better clarity
            text_with_pauses = " ... ".join(text.split())
            engine.say(text_with_pauses)
            engine.runAndWait()
            speech_queue.task_done()
        except:
            continue

def speak_text(text):
    """
    Função que adiciona texto a lista de fala
    """
    global last_spoken_time, last_spoken_text
    current_time = time.time()
    
    if (current_time - last_spoken_time >= 2.5 and text != last_spoken_text):
        speech_queue.put(text)
        last_spoken_time = current_time
        last_spoken_text = text

# Start the speech worker thread
speech_thread = Thread(target=speak_worker, daemon=True)
speech_thread.start()

# Load model and initialize video capture
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

# palavras que serão detectadas e comparadas de acordo com a ordem
labels_dict = {0: 'Oi', 1: 'B', 2: 'C', 3:'D', 4:'E'}

# Create more natural speaking phrases in Portuguese
def format_speech_text(detected_signs):
    if len(detected_signs) == 1:
        return f"{detected_signs[0]}"
    elif len(detected_signs) == 2:
        return f"{detected_signs[0]}{detected_signs[1]}"
    elif len(detected_signs) > 2:
        signs_text = "".join(detected_signs[:-1]) + f"{detected_signs[-1]}"
        return f"{signs_text}"
    return ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detected_signs = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Reset data collection for each hand
                data_aux = []
                x_ = []
                y_ = []

                # Collect coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Calculate normalized coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Calculate bounding box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    # Make prediction
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    detected_signs.append(predicted_character)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                               cv2.LINE_AA)
                except Exception as e:
                    print(f"Erro de previsão para mão {hand_idx}: {str(e)}")
                    continue

            # Format and queue speech text
            if detected_signs:
                speech_text = format_speech_text(detected_signs)
                speak_text(speech_text)

        # Show frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    speech_thread_running = False
    speech_thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()