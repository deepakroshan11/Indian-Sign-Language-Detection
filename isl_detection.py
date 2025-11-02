# isl_detection.py - ENHANCED VERSION
# NEW FEATURE: Letter + Confidence overlay on video feed
# Displays detected letter with accuracy in top-left corner
# No changes to emotion detection, dashboard, or graph functionality

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import time
import copy
import itertools
import os
import threading
from collections import deque, defaultdict, Counter
from multiprocessing import Queue, Value
import ctypes

from tensorflow.keras.models import load_model
from fer import FER
from gtts import gTTS
import pygame

# -------------------------------
# ---------- CONFIG -------------
# -------------------------------

ISL_MODEL_PATH = "model.h5"
CAM_INDEX = 0

BASE_FER_WEIGHT = 0.65
BASE_LANDMARK_WEIGHT = 0.35
SMOOTHING_FRAMES = 15
MIN_CONF_TO_SHOW = 0.25
EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral"]

BUFFER_SIZE = 12
CONF_THRESHOLD = 0.8
STABILITY_THRESHOLD = 0.7
SENTENCE_DELAY = 2.0
SPACE_THRESHOLD = 15

# TTS Settings
ENABLE_AUTO_TTS = True
TTS_LANGUAGE = "en"

# NEW: Display Settings for Letter Overlay
SHOW_LETTER_OVERLAY = True  # Enable/disable letter display on video
OVERLAY_COLOR_LETTER = (0, 255, 255)  # Yellow for letter (BGR)
OVERLAY_COLOR_CONF = (255, 255, 0)    # Cyan for confidence (BGR)
OVERLAY_POSITION = (20, 50)           # Top-left position
OVERLAY_FONT_SCALE = 1.5              # Font size
OVERLAY_THICKNESS = 3                 # Text thickness

COMMON_WORDS = [
    "HELLO", "HELP", "PLEASE", "THANK", "YOU", "YES", "NO", "GOOD", "BAD",
    "MORNING", "AFTERNOON", "EVENING", "NIGHT", "TODAY", "TOMORROW", "WATER",
    "FOOD", "HOME", "SCHOOL", "WORK", "HAPPY", "SAD", "SORRY", "WELCOME"
]

# -------------------------------
# --------- SHARED STATE --------
# -------------------------------

class SharedState:
    """Shared memory structure for inter-process communication"""
    def __init__(self):
        self.frame_width = Value(ctypes.c_int, 640)
        self.frame_height = Value(ctypes.c_int, 480)
        self.frame_ready = Value(ctypes.c_bool, False)
        self.ui_queue = Queue(maxsize=2)
        self.command_queue = Queue(maxsize=10)
        self.fps = Value(ctypes.c_double, 0.0)
        self.processing_fps = Value(ctypes.c_double, 0.0)

# -------------------------------
# --------- TTS ENGINE ----------
# -------------------------------

try:
    pygame.mixer.init()
    print("✅ TTS engine initialized")
except Exception as e:
    print(f"⚠️  TTS init warning: {e}")

speaking_word = ""
pause_audio = False

def speak_text(text, lang=TTS_LANGUAGE):
    """Non-blocking TTS using gTTS + pygame"""
    global speaking_word
    if not text or not text.strip():
        return

    def _speak():
        global pause_audio, speaking_word
        filename = None
        try:
            filename = f"voice_{int(time.time()*1000)}_{threading.get_ident()}.mp3"
            tts = gTTS(text=text, lang=lang)
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
            
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.1)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                    else:
                        print(f"⚠️  Could not delete {filename}")
        except Exception as e:
            print(f"TTS error: {e}")
            speaking_word = ""

    threading.Thread(target=_speak, daemon=True).start()

# -------------------------------
# --------- UTILITIES -----------
# -------------------------------

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(lm.x * image_width), image_width - 1),
             min(int(lm.y * image_height), image_height - 1)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp))) if temp else 1
    return [n / max_value for n in temp]

def apply_clahe(frame):
    """CLAHE for better lighting normalization"""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame

def compute_brightness(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    return float(np.mean(yuv[:,:,0]))

def extract_face_features(face_landmarks, img_shape):
    try:
        pts = np.array(face_landmarks, dtype=np.float32)
        h, w = img_shape[:2]
        face_width = max(np.max(pts[:,0]) - np.min(pts[:,0]), 1.0)

        left_corner = pts[78]; right_corner = pts[308]
        top_lip = pts[13]; bottom_lip = pts[14]
        mouth_left = left_corner; mouth_right = right_corner

        mouth_w = np.linalg.norm(mouth_right - mouth_left)
        mouth_h = np.linalg.norm(bottom_lip - top_lip)
        mouth_ratio = mouth_w / (mouth_h + 1e-6)
        mouth_open = mouth_h / face_width

        left_eye_open = np.linalg.norm(pts[159] - pts[145]) / face_width
        right_eye_open = np.linalg.norm(pts[386] - pts[374]) / face_width
        eye_open = (left_eye_open + right_eye_open) / 2.0

        left_brow_gap = np.linalg.norm(pts[70] - pts[159]) / face_width
        right_brow_gap = np.linalg.norm(pts[300] - pts[386]) / face_width
        brow_gap = (left_brow_gap + right_brow_gap) / 2.0

        lip_center_y = (top_lip[1] + bottom_lip[1]) / 2.0
        corner_upness = ((mouth_left[1] - lip_center_y) + (mouth_right[1] - lip_center_y)) / 2.0
        mouth_corner_dir = -corner_upness / face_width

        chin = pts[152] if pts.shape[0] > 152 else np.array([np.mean(pts[:,0]), np.max(pts[:,1])])
        jaw_tension = np.linalg.norm(top_lip - chin) / face_width

        return {
            "mouth_ratio": float(mouth_ratio),
            "mouth_open": float(mouth_open),
            "eye_open": float(eye_open),
            "brow_gap": float(brow_gap),
            "mouth_corner_dir": float(mouth_corner_dir),
            "jaw_tension": float(jaw_tension),
        }
    except Exception:
        return {}

def landmark_scores_from_features(feat):
    if not feat:
        return {e: 0.0 for e in EMOTIONS[:-1]} | {"neutral": 1.0}

    mouth_ratio = feat.get("mouth_ratio", 1.0)
    mouth_open = feat.get("mouth_open", 0.0)
    corner = feat.get("mouth_corner_dir", 0.0)
    
    happy = 0.0
    if corner > 0.01:
        happy = min(1.0, (corner / 0.08) + max(0.0, (mouth_ratio - 1.2)/0.8))
    happy = max(0.0, min(1.0, happy))

    sad = 0.0
    if corner < -0.005 or mouth_ratio < 1.05:
        sad = min(1.0, (abs(min(0.0, corner)) / 0.06) + max(0.0, (1.05 - mouth_ratio)/0.4))
    sad = max(0.0, min(1.0, sad))

    brow_gap = feat.get("brow_gap", 0.08)
    eye_open = feat.get("eye_open", 0.08)
    jaw_t = feat.get("jaw_tension", 0.02)
    angry = 0.0
    if brow_gap < 0.06:
        angry += (0.06 - brow_gap) / 0.06 * 0.6
    if eye_open < 0.06:
        angry += (0.06 - eye_open) / 0.06 * 0.3
    if jaw_t > 0.035:
        angry += min(0.3, (jaw_t - 0.035) / 0.05)
    angry = max(0.0, min(1.0, angry))

    surprise = 0.0
    if brow_gap > 0.12:
        surprise += min(1.0, (brow_gap - 0.12)/0.08) * 0.6
    if feat.get("eye_open",0) > 0.12:
        surprise += min(1.0, (feat.get("eye_open") - 0.12)/0.15) * 0.4
    if feat.get("mouth_open",0) > 0.04:
        surprise += min(0.4, (feat.get("mouth_open") - 0.04) / 0.1)
    surprise = max(0.0, min(1.0, surprise))

    other_max = max(happy, sad, angry, surprise)
    neutral = max(0.0, 1.0 - other_max)

    return {
        "happy": float(happy),
        "sad": float(sad),
        "angry": float(angry),
        "surprise": float(surprise),
        "neutral": float(neutral)
    }

def get_word_suggestions(partial_word, word_freq):
    if not partial_word:
        return []
    
    partial = partial_word.upper()
    matches = []
    
    for word, freq in word_freq.most_common(20):
        if word.startswith(partial) and word != partial:
            matches.append((word, freq))
    
    for word in COMMON_WORDS:
        if word.startswith(partial) and word not in [m[0] for m in matches]:
            matches.append((word, 0))
    
    matches.sort(key=lambda x: (-x[1], x[0]))
    return [m[0] for m in matches[:3]]

def clear_camera_buffer(cap, num_frames=5):
    """Clear camera buffer to reduce lag"""
    for _ in range(num_frames):
        cap.read()

# -------------------------------
# ---- DRAW HAND LANDMARKS ------
# -------------------------------

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw colored hand skeleton overlay"""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    landmark_style = mp_drawing.DrawingSpec(
        color=(0, 255, 0),  # Green dots
        thickness=3,
        circle_radius=4
    )
    connection_style = mp_drawing.DrawingSpec(
        color=(0, 0, 255),  # Red lines
        thickness=2
    )
    
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        landmark_style,
        connection_style
    )

# -------------------------------
# --- NEW: DRAW LETTER OVERLAY --
# -------------------------------

def draw_letter_overlay(frame, letter, confidence):
    """
    Draw detected letter and confidence on video frame
    Similar to the image shown - displays in top-left corner
    """
    if not SHOW_LETTER_OVERLAY:
        return frame
    
    if letter and confidence > 0:
        # Format text: "S (1.00)"
        text_letter = f"{letter}"
        text_conf = f"({confidence:.2f})"
        
        x, y = OVERLAY_POSITION
        
        # Add semi-transparent background for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-40), (x+180, y+20), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw letter in yellow
        cv2.putText(
            frame, 
            text_letter, 
            (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            OVERLAY_FONT_SCALE, 
            OVERLAY_COLOR_LETTER, 
            OVERLAY_THICKNESS,
            cv2.LINE_AA
        )
        
        # Draw confidence in cyan next to letter
        text_width = cv2.getTextSize(text_letter, cv2.FONT_HERSHEY_SIMPLEX, 
                                     OVERLAY_FONT_SCALE, OVERLAY_THICKNESS)[0][0]
        cv2.putText(
            frame, 
            text_conf, 
            (x + text_width + 10, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            OVERLAY_FONT_SCALE * 0.7,  # Slightly smaller
            OVERLAY_COLOR_CONF, 
            OVERLAY_THICKNESS - 1,
            cv2.LINE_AA
        )
    
    return frame

# -------------------------------
# ------- CORE PROCESSOR --------
# -------------------------------

def core_processing_engine(shared_state):
    """Enhanced processing engine with letter overlay feature"""
    
    print("🚀 Starting ENHANCED core processing engine...")
    print("✨ Features: Hand overlay, Letter display, Auto-TTS, Emotion detection")
    
    # Load models
    try:
        model = load_model(ISL_MODEL_PATH)
        print("✅ ISL model loaded")
    except Exception as e:
        print(f"❌ Failed to load ISL model: {e}")
        model = None
    
    alphabet = list(string.ascii_uppercase)
    emotion_detector = FER(mtcnn=False)
    
    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    
    # State variables
    buffer = []
    current_word = []
    sentence = []
    no_hand_frames = 0
    last_letter = None
    letter_confidence_buffer = deque(maxlen=5)
    hand_position_buffer = deque(maxlen=5)
    word_frequency = Counter()
    
    # NEW: Track current detected letter and confidence for overlay
    current_detected_letter = ""
    current_confidence = 0.0
    
    # Emotion tracking
    emotion_history = deque(maxlen=SMOOTHING_FRAMES)
    current_emotion = "neutral"
    emotion_scores = {e: 0.0 for e in EMOTIONS}
    emotion_timeline = deque(maxlen=100)
    last_emotion = "neutral"
    
    # Performance tracking
    frame_count = 0
    last_fps_update = time.time()
    
    # Stats
    stats = {
        "letters_detected": 0,
        "words_formed": 0,
        "sentences_formed": 0,
        "emotion_changes": 0,
        "session_start": time.time()
    }
    
    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    print("🎥 Clearing camera buffer...")
    clear_camera_buffer(cap, 10)
    print("✅ Camera ready")
    
    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.6, 
                        min_tracking_confidence=0.6) as hands, \
         mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                         min_detection_confidence=0.5, 
                         min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            loop_start = time.time()
            
            # Command handling
            try:
                while not shared_state.command_queue.empty():
                    cmd = shared_state.command_queue.get_nowait()
                    
                    if cmd['action'] == 'reset':
                        buffer = []
                        current_word = []
                        sentence = []
                        no_hand_frames = 0
                        last_letter = None
                        letter_confidence_buffer.clear()
                        hand_position_buffer.clear()
                        current_detected_letter = ""
                        current_confidence = 0.0
                        print("🔄 Reset executed")
                    
                    elif cmd['action'] == 'backspace':
                        if current_word:
                            removed = current_word.pop()
                            print(f"⌫ Backspace: removed '{removed}'")
                        elif sentence:
                            removed_word = sentence.pop()
                            print(f"⌫ Backspace: removed word '{removed_word}'")
                    
                    elif cmd['action'] == 'accept_suggestion':
                        suggestion = cmd.get('word', '')
                        if suggestion:
                            current_word = list(suggestion)
                            print(f"✅ Accepted suggestion: {suggestion}")
                    
                    elif cmd['action'] == 'speak':
                        text_to_speak = cmd.get('text', '')
                        if not text_to_speak:
                            if sentence:
                                text_to_speak = " ".join(sentence)
                            elif current_word:
                                text_to_speak = "".join(current_word)
                        
                        if text_to_speak:
                            print(f"🔊 Speaking: {text_to_speak}")
                            speak_text(text_to_speak)
                    
                    elif cmd['action'] == 'stop':
                        print("🛑 Stop signal received")
                        cap.release()
                        return
            except:
                pass
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            hand_results = hands.process(frame_rgb)
            detected_letter = ""
            hand_detected = False
            
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                no_hand_frames = 0
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand skeleton
                    draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Check stability
                    wrist = hand_landmarks.landmark[0]
                    pos = np.array([wrist.x * frame.shape[1], wrist.y * frame.shape[0]])
                    hand_position_buffer.append(pos)
                    
                    is_stable = False
                    if len(hand_position_buffer) >= 3:
                        positions = np.array(hand_position_buffer)
                        variance = np.var(positions, axis=0)
                        stability = 1.0 / (1.0 + np.mean(variance) / 100)
                        is_stable = stability > STABILITY_THRESHOLD
                    
                    try:
                        lm_list = calc_landmark_list(frame, hand_landmarks)
                        pre_list = pre_process_landmark(lm_list)
                        df = pd.DataFrame(pre_list).transpose()
                        
                        if model is not None and is_stable:
                            pred = model.predict(df, verbose=0)
                            prob = float(np.max(pred))
                            letter_confidence_buffer.append(prob)
                            
                            avg_conf = np.mean(letter_confidence_buffer) if letter_confidence_buffer else 0
                            
                            # NEW: Update overlay display even before confirmation
                            predicted_letter = alphabet[int(np.argmax(pred))]
                            current_detected_letter = predicted_letter
                            current_confidence = avg_conf
                            
                            if avg_conf >= CONF_THRESHOLD:
                                detected_letter = predicted_letter
                                buffer.append(detected_letter)
                                if len(buffer) > BUFFER_SIZE:
                                    buffer.pop(0)
                                if buffer.count(detected_letter) > BUFFER_SIZE // 2:
                                    if (last_letter is None) or (detected_letter != last_letter):
                                        current_word.append(detected_letter)
                                        last_letter = detected_letter
                                        stats["letters_detected"] += 1
                                        print(f"✍️  Letter confirmed: {detected_letter} ({avg_conf:.2f})")
                                        buffer.clear()
                                        letter_confidence_buffer.clear()
                    except Exception as e:
                        pass
            else:
                no_hand_frames += 1
                letter_confidence_buffer.clear()
                current_detected_letter = ""
                current_confidence = 0.0
                
                if no_hand_frames >= SPACE_THRESHOLD:
                    if current_word:
                        word = "".join(current_word)
                        sentence.append(word)
                        word_frequency[word] += 1
                        stats["words_formed"] += 1
                        print(f"📝 Word formed: {word}")
                        
                        if ENABLE_AUTO_TTS:
                            speak_text(word)
                        
                        current_word = []
                        last_letter = None
                    no_hand_frames = 0
            
            # NEW: Draw letter overlay on frame BEFORE emotion processing
            frame = draw_letter_overlay(frame, current_detected_letter, current_confidence)
            
            # Emotion detection (unchanged)
            frame_clahe = apply_clahe(frame)
            brightness = compute_brightness(frame_clahe)
            bright_scale = np.clip((brightness / 255.0)*1.2, 0.5, 1.0)
            fer_weight = BASE_FER_WEIGHT * bright_scale
            landmark_weight = max(0.0, 1.0 - fer_weight)
            
            fer_probs = {}
            try:
                fer_res = emotion_detector.detect_emotions(frame_clahe)
                if fer_res:
                    fer_probs = fer_res[0]["emotions"]
                else:
                    fer_probs = {"angry":0,"happy":0,"sad":0,"surprise":0,"neutral":1.0}
            except:
                fer_probs = {"angry":0,"happy":0,"sad":0,"surprise":0,"neutral":1.0}
            
            fer_subset = {
                "happy": fer_probs.get("happy",0.0),
                "sad": fer_probs.get("sad",0.0),
                "angry": fer_probs.get("angry",0.0),
                "surprise": fer_probs.get("surprise",0.0),
                "neutral": fer_probs.get("neutral",0.0)
            }
            
            mesh_res = face_mesh.process(frame_rgb)
            landmark_scores = {k:0.0 for k in EMOTIONS}
            if mesh_res.multi_face_landmarks:
                landmarks = mesh_res.multi_face_landmarks[0]
                face_landmarks = calc_landmark_list(frame, landmarks)
                feats = extract_face_features(face_landmarks, frame.shape)
                landmark_scores = landmark_scores_from_features(feats)
            
            fused = {}
            for emo in EMOTIONS:
                f = fer_subset.get(emo, 0.0)
                l = landmark_scores.get(emo, 0.0)
                fused_score = fer_weight * f + landmark_weight * l
                fused[emo] = float(np.clip(fused_score, 0.0, 1.0))
            
            emotion_history.append(fused)
            avg_emotions = defaultdict(float)
            for d in emotion_history:
                for k,v in d.items():
                    avg_emotions[k] += v
            if len(emotion_history) > 0:
                for k in avg_emotions:
                    avg_emotions[k] = avg_emotions[k] / len(emotion_history)
            
            if avg_emotions:
                max_emo = max(avg_emotions, key=avg_emotions.get)
                max_val = avg_emotions[max_emo]
                dominant = max_emo if max_val >= MIN_CONF_TO_SHOW else "neutral"
            else:
                dominant = "neutral"
                avg_emotions = {k:0.0 for k in EMOTIONS}
                avg_emotions["neutral"] = 1.0
            
            if current_emotion != dominant and dominant != last_emotion:
                stats["emotion_changes"] += 1
                print(f"😊 Emotion changed: {current_emotion} → {dominant}")
                last_emotion = current_emotion
            
            current_emotion = dominant
            emotion_scores = dict(avg_emotions)
            
            emotion_timeline.append({
                "time": time.time(),
                "emotion": dominant,
                "scores": dict(avg_emotions)
            })
            
            # Calculate FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_update >= 0.5:
                fps = frame_count / (now - last_fps_update)
                shared_state.processing_fps.value = fps
                frame_count = 0
                last_fps_update = now
            
            # Get suggestions
            current_display = "".join(current_word)
            suggestions = get_word_suggestions(current_display, word_frequency)
            
            # Prepare data packet for UI
            try:
                _, buffer_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer_img.tobytes()
                
                data_packet = {
                    "frame": frame_bytes,
                    "frame_shape": frame.shape,
                    "current_word": current_display,
                    "sentence": " ".join(sentence),
                    "suggestions": suggestions,
                    "emotion": current_emotion,
                    "emotion_scores": emotion_scores,
                    "emotion_timeline": list(emotion_timeline)[-20:],
                    "stats": stats.copy(),
                    "detected_letter": detected_letter,
                    "hand_detected": hand_detected,
                    "fps": shared_state.processing_fps.value,
                    "timestamp": time.time(),
                    "speaking": speaking_word,
                    # NEW: Send overlay info for dashboard display too
                    "overlay_letter": current_detected_letter,
                    "overlay_confidence": current_confidence
                }
                
                if shared_state.ui_queue.full():
                    try:
                        shared_state.ui_queue.get_nowait()
                    except:
                        pass
                
                shared_state.ui_queue.put_nowait(data_packet)
            except:
                pass
            
            # Dynamic sleep
            loop_time = time.time() - loop_start
            target_frame_time = 0.033
            if loop_time < target_frame_time:
                time.sleep(target_frame_time - loop_time)
            
            if frame_count % 30 == 0:
                clear_camera_buffer(cap, 2)
    
    cap.release()
    print("✅ Enhanced core processor stopped")

if __name__ == "__main__":
    shared_state = SharedState()
    core_processing_engine(shared_state)