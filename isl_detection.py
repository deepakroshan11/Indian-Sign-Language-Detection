# isl_detection_final.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import time
import copy
import itertools
import threading
import os
from collections import deque
from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame
from hdfs import InsecureClient
import tempfile

# -------------------------------
# HDFS Model Loader
# -------------------------------
def load_hdfs_model(hdfs_url, hdfs_user, hdfs_path):
    """Download Keras model from HDFS and load it."""
    client = InsecureClient(hdfs_url, user=hdfs_user)
    with client.read(hdfs_path) as reader:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='emotion_model.h5')
        tmp_file.write(reader.read())
        tmp_file.close()
        model = load_model(tmp_file.name, compile=False)
        os.remove(tmp_file.name)
    return model

# -------------------------------
# Load Models & Setup
# -------------------------------
# Gesture model (your existing A-Z model)
gesture_model = load_model("model.h5")
alphabet = list(string.ascii_uppercase)

# Emotion model from HDFS
HDFS_URL = 'http://<namenode_host>:50070'  # Replace with your HDFS namenode URL
HDFS_USER = '<hdfs_user>'                  # Replace with your HDFS username
HDFS_MODEL_PATH = '/path/to/emotion_model.h5.hdfs'  # Replace with HDFS model path

emotion_model = load_hdfs_model(HDFS_URL, HDFS_USER, HDFS_MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face detector (OpenCV Haar)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# Parameters
# -------------------------------
BUFFER_SIZE = 12
CONF_THRESHOLD = 0.8
SENTENCE_DELAY = 2.0
SPACE_THRESHOLD = 15
EMOTION_POLL_SEC = 0.18
EMOTION_HISTORY_LEN = 7
GESTURE_ROI = (100, 100, 400, 400)

# -------------------------------
# Shared variables
# -------------------------------
buffer = []
current_word = []
sentence = []
last_sentence_time = time.time()
no_hand_frames = 0
sentence_spoken = False
pause_audio = False
fps = 0
prev_time = time.time()
speaking_word = ""
blink_flag = True
last_letter = None
last_letter_time = time.time()
emotion_history = deque(maxlen=EMOTION_HISTORY_LEN)
current_emotion = "Neutral"
_emotion_lock = threading.Lock()
frame_queue = deque(maxlen=1)

# -------------------------------
# Init TTS (pygame)
# -------------------------------
pygame.mixer.init()

def speak_text(text):
    global speaking_word
    if not text or not text.strip():
        return

    def _speak():
        global pause_audio, speaking_word
        try:
            filename = f"voice_{int(time.time()*1000)}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            speaking_word = text
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if pause_audio:
                    pygame.mixer.music.pause()
                    while pause_audio:
                        time.sleep(0.1)
                    pygame.mixer.music.unpause()
                pygame.time.Clock().tick(10)
            speaking_word = ""
            os.remove(filename)
        except Exception as e:
            print("TTS error:", e)
            speaking_word = ""

    threading.Thread(target=_speak, daemon=True).start()

# -------------------------------
# Landmark processing for gesture
# -------------------------------
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp)))
    if max_value == 0:
        return [0]*len(temp)
    return [n / max_value for n in temp]

# -------------------------------
# Emotion detection worker
# -------------------------------
def emotion_worker():
    global current_emotion
    while True:
        if frame_queue:
            frame = frame_queue[-1]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
            label = None
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                (x,y,w,h) = faces[0]
                face = gray[y:y+h, x:x+w]
                try:
                    face = cv2.resize(face, (48,48))
                    face = face.astype("float32") / 255.0
                    face = np.expand_dims(face, axis=0)
                    face = np.expand_dims(face, axis=-1)
                    preds = emotion_model.predict(face, verbose=0)[0]
                    label = emotion_labels[np.argmax(preds)]
                except:
                    label = None
                try:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    if label:
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                except:
                    pass
            if label:
                emotion_history.append(label)
                smoothed = max(set(emotion_history), key=emotion_history.count)
                with _emotion_lock:
                    current_emotion = smoothed
            else:
                with _emotion_lock:
                    current_emotion = "Neutral"
        time.sleep(EMOTION_POLL_SEC)

# -------------------------------
# Utility & persistence
# -------------------------------
def reset_all():
    global buffer, current_word, sentence, last_sentence_time, no_hand_frames, sentence_spoken, last_letter
    buffer = []
    current_word = []
    sentence = []
    last_sentence_time = time.time()
    no_hand_frames = 0
    sentence_spoken = False
    last_letter = None
    print("✅ Reset complete")

def save_sentence(text):
    try:
        with open("sentences.txt", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
    except Exception as e:
        print("Could not save sentence:", e)

# -------------------------------
# Main loop
# -------------------------------
def main():
    global buffer, current_word, sentence, last_sentence_time, no_hand_frames
    global sentence_spoken, pause_audio, fps, prev_time, speaking_word, blink_flag, last_letter, last_letter_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    t = threading.Thread(target=emotion_worker, daemon=True)
    t.start()

    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            frame_queue.append(frame.copy())

            # Gesture detection
            label = ""
            if results.multi_hand_landmarks:
                no_hand_frames = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))
                    try:
                        lm_list = calc_landmark_list(frame, hand_landmarks)
                        pre_list = pre_process_landmark(lm_list)
                        df = pd.DataFrame(pre_list).transpose()
                        pred = gesture_model.predict(df, verbose=0)[0]
                        prob = np.max(pred)
                        if prob >= CONF_THRESHOLD:
                            label = alphabet[np.argmax(pred)]
                            buffer.append(label)
                            if len(buffer) > BUFFER_SIZE:
                                buffer.pop(0)
                            if buffer.count(label) > BUFFER_SIZE // 2:
                                if (last_letter is None) or (label != last_letter):
                                    current_word.append(label)
                                    last_letter = label
                                    last_letter_time = now
                                    buffer.clear()
                    except:
                        pass
            else:
                no_hand_frames += 1
                if no_hand_frames >= SPACE_THRESHOLD:
                    if current_word:
                        word = "".join(current_word)
                        sentence.append(word)
                        print("📝 Word formed:", word)
                        speak_text(word)
                        save_sentence(" ".join(sentence))
                        current_word = []
                        last_sentence_time = now
                        sentence_spoken = False
                        last_letter = None
                    no_hand_frames = 0

            # Sentence TTS
            if sentence and not current_word and not sentence_spoken:
                if now - last_sentence_time >= SENTENCE_DELAY:
                    full_text = " ".join(sentence)
                    print("📢 Sentence:", full_text)
                    speak_text(full_text)
                    save_sentence(full_text)
                    sentence_spoken = True

            # Display current word & sentence
            cv2.putText(frame, "".join(current_word), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
            if sentence:
                last_word = sentence[-1]
                sentence_prefix = " ".join(sentence[:-1])
                if speaking_word == last_word:
                    blink_flag = not blink_flag
                    color = (0,255,255) if blink_flag else (0,0,0)
                else:
                    color = (0,255,255)
                display_sentence = (sentence_prefix + " " + last_word).strip()
                cv2.putText(frame, display_sentence, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            with _emotion_lock:
                emotion_to_show = current_emotion
            cv2.putText(frame, f"Emotion: {emotion_to_show}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            x1,y1,x2,y2 = GESTURE_ROI
            cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,200), 1)
            cv2.imshow("ISL Translator + Emotion", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                reset_all()
            elif key == ord('p'):
                pause_audio = True
            elif key == ord('r'):
                pause_audio = False
            elif key == ord('b'):
                if current_word:
                    removed = current_word.pop()
                    print(f"⌫ Backspace removed: {removed}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
