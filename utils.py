import cv2
import numpy as np
import pickle
import os
import datetime
import time 
from collections import deque
import gc 
import io 

# --- New Imports for Security and Deep Learning ---
try:
    from cryptography.fernet import Fernet
except ImportError:
    print("Warning: 'cryptography' not installed. Install with: pip install cryptography")
    Fernet = None

try:
    import tensorflow as tf
    # Using the standard Keras loader for .h5 compatibility
    from tensorflow.keras.models import load_model 
    import h5py 
except ImportError:
    print("Warning: 'tensorflow' or 'h5py' not installed. Mask/Liveness models cannot be loaded.")
    tf = None
    load_model = None
    h5py = None

# --- Global Variables ---
mask_detector_model = None
liveness_detector_model = None 
ENCRYPTION_KEY = None
KEY_FILE = 'encrypted_data/secret.key'

BLINK_HISTORY_LEN = 15
landmark_history = deque(maxlen=BLINK_HISTORY_LEN)

# --- 1. ENCODING MANAGEMENT ---

def generate_and_save_key(key_path=KEY_FILE):
    if Fernet is None: return None
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    key = Fernet.generate_key()
    with open(key_path, "wb") as key_file:
        key_file.write(key)
    return key

def load_encryption_key(key_path=KEY_FILE):
    global ENCRYPTION_KEY
    if Fernet is None: return None
    if not os.path.exists(key_path):
        ENCRYPTION_KEY = generate_and_save_key(key_path)
    else:
        with open(key_path, "rb") as key_file:
            ENCRYPTION_KEY = key_file.read()
    return ENCRYPTION_KEY

load_encryption_key() 

def load_encodings(encodings_file='encrypted_data/face_encodings.pkl'):
    global ENCRYPTION_KEY
    if Fernet is None or ENCRYPTION_KEY is None:
        return [], []
    try:
        if not os.path.exists(encodings_file):
            return [], []
        f = Fernet(ENCRYPTION_KEY)
        with open(encodings_file, 'rb') as ef:
            encrypted_data = ef.read()
            decrypted_data = f.decrypt(encrypted_data)
            known_encodings, known_names = pickle.loads(decrypted_data)
        known_encodings = np.array(known_encodings)
        return known_encodings, known_names
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def save_encodings(known_encodings, known_names, encodings_file='encrypted_data/face_encodings.pkl'):
    global ENCRYPTION_KEY
    if Fernet is None or ENCRYPTION_KEY is None: return False
    f = Fernet(ENCRYPTION_KEY)
    pickled_data = pickle.dumps((known_encodings, known_names))
    encrypted_data = f.encrypt(pickled_data)
    os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
    with open(encodings_file, 'wb') as ef:
        ef.write(encrypted_data)
    return True

# --- 2. MODEL LOADING (Updated Fix) ---

def load_mask_detector(model_path="models/mask_detector.h5"):
    """ Loads the Masking Detection Model using the most compatible Keras 3 method. """
    global mask_detector_model
    if tf is None: return False

    try:
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return False
            
        # 1. Clear any previous session remnants
        tf.keras.backend.clear_session()
        
        # 2. Use the 'compile=False' flag to avoid loading optimizer weights 
        # (This is usually where the 'utf-8' error happens)
        mask_detector_model = tf.keras.models.load_model(model_path, compile=False)
        
        print("✅ Mask detection model loaded successfully (Inference Mode).")
        return True
        
    except Exception as e:
        print(f"Error loading mask model: {e}")
        # If it STILL fails, we check if it's a path issue
        abs_path = os.path.abspath(model_path)
        print(f"Attempted Absolute Path: {abs_path}")
        mask_detector_model = False 
        return False

def load_liveness_detector(model_path="models/liveness_detector.h5"):
    print("Using EAR-based Liveness detection.")
    return True

# --- 3. LIVENESS DETECTION (EAR-based) ---

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def check_blink_sequence(landmarks_sequence, ear_threshold=0.20):
    ears = []
    for ls in landmarks_sequence:
        if ls is None:
            ears.append(0.0)
            continue
        left_eye, right_eye = ls.get('left_eye'), ls.get('right_eye')
        if left_eye and right_eye:
            ears.append((eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0)
        else:
            ears.append(0.0)

    if len(ears) < 3: return False
    for i in range(1, len(ears)-1):
        if (ears[i] < ear_threshold and ears[i-1] >= ear_threshold and ears[i+1] >= ear_threshold):
            return True
    return False

def detect_liveness_ear(current_landmarks):
    global landmark_history
    landmark_history.append(current_landmarks)
    is_live = check_blink_sequence(landmark_history, ear_threshold=0.22)
    return is_live, 1.0

# --- 4. MASK DETECTION ---

def detect_mask(face_image_roi):
    global mask_detector_model
    if not mask_detector_model:
        return False, 0.5 
    
    try:
        face_input = np.expand_dims(face_image_roi, axis=0) 
        prediction = mask_detector_model.predict(face_input, verbose=0)[0]
        
        # Binary Classification: [Masked, Unmasked]
        MASK_INDEX = 0 
        is_masked = prediction[MASK_INDEX] > 0.5
        confidence = prediction[MASK_INDEX] if is_masked else prediction[1]
        
        return is_masked, float(confidence)
    except Exception as e:
        print(f"Inference error: {e}")
        return False, 0.5 

# --- 5. ATTENDANCE LOGGING ---

def log_attendance(name, status, log_file='attendance/attendance_log.csv'):
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{dt_string},{name},{status}\n"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    write_header = not os.path.exists(log_file)
    with open(log_file, 'a') as f:
        if write_header: f.write("Timestamp,Name,Status\n")
        f.write(log_entry)
    return True

# --- 6. UI DRAWING ---

def draw_info_on_frame(frame, face_location, name, is_live, is_masked):
    top, right, bottom, left = face_location
    color = (0, 255, 0)
    status_text = name
    
    if name == "Unknown":
        color = (255, 0, 0)
    else:
        if not is_live:
            status_text += " | SPOOF! ❌"
            color = (0, 0, 255)
        elif is_masked:
            status_text += " | MASKED 😷"
            color = (255, 165, 0)
            if mask_detector_model is False:
                 status_text = name + " | MASK CHECK ERROR"
        else:
            status_text += " | PRESENT ✅"
            color = (0, 255, 0)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, status_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    return frame

# --- 7. MASK OVERLAY ---

def overlay_mask_on_face(img, face_landmarks, mask_path):
    try:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None: return img
        nose_bridge, chin = face_landmarks.get('nose_bridge'), face_landmarks.get('chin')
        if not nose_bridge or not chin: return img

        face_width = chin[-1][0] - chin[0][0]
        mask_width = int(face_width * 1.5)
        mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
        resized_mask = cv2.resize(mask_img, (mask_width, mask_height))

        x1, y1 = nose_bridge[2][0] - mask_width // 2, nose_bridge[2][1] - mask_height // 3
        x2, y2 = x1 + mask_width, y1 + mask_height
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        
        roi = img[y1:y2, x1:x2]
        resized_mask = resized_mask[:roi.shape[0], :roi.shape[1]]

        if resized_mask.shape[2] == 4:
            alpha_s = resized_mask[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = (alpha_s * resized_mask[:, :, c] + (1.0 - alpha_s) * roi[:, :, c])
        return img
    except Exception:
        return img