# ============================================================
#  REAL-TIME EMOTION DETECTION
#  Face Detection : OpenCV Haar Cascade (reliable face crop)
#  Emotion Model  : Mini VGG CNN (FER2013)
#  Features       : Single emotion display, smoothed predictions
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import collections
import time
import os

# Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Config ───────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\RIA MALHOTRA\OneDrive\Desktop\New folder\FacialEmotionRecognition\models\best_emotion_model.keras"
SMOOTH_N    = 20        # Frames to average — higher = more stable
SKIP_FRAMES = 3         # Predict every N frames — reduces flickering

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJIS   = ["😠",    "🤢",      "😨",   "😄",    "😐",      "😢",  "😲"     ]

# One color per emotion (BGR)
EMOTION_COLORS = {
    "Angry"   : (0,   0,   220),
    "Disgust" : (130, 0,   130),
    "Fear"    : (0,   140, 255),
    "Happy"   : (0,   200, 0  ),
    "Neutral" : (130, 130, 130),
    "Sad"     : (200, 80,  0  ),
    "Surprise": (0,   200, 255),
}

# ── Load Emotion Model ────────────────────────────────────────
print("[INFO] Loading emotion model...")
emotion_model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded!")

# ── Load Face Detector (Haar Cascade — reliable face crop) ───
# Much better than YOLO person detection for getting face region
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("[INFO] Face detector ready!")

# ── Webcam ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Camera started! Press Q to quit.\n")

# ── Smoothing buffers ─────────────────────────────────────────
emotion_buffer = collections.deque(maxlen=SMOOTH_N)
prob_buffer    = collections.deque(maxlen=SMOOTH_N)

# Cached stable result
stable_emotion = "Neutral"
stable_color   = EMOTION_COLORS["Neutral"]
stable_conf    = 0.0

frame_count    = 0
fps            = 0
fps_time       = time.time()

# ── Main Loop ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect Faces ─────────────────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)    # ignore tiny detections
    )

    for (x, y, w, h) in faces:

        # ── Only predict every SKIP_FRAMES frames ─────────────
        if frame_count % SKIP_FRAMES == 0:

            # Extract face region — grayscale, resize to 48x48
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_input   = face_resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

            # Predict
            probs = emotion_model.predict(face_input, verbose=0)[0]

            # Add to smoothing buffer
            emotion_buffer.append(np.argmax(probs))
            prob_buffer.append(probs)

            # Stable emotion = most common in last SMOOTH_N frames
            stable_id      = collections.Counter(emotion_buffer).most_common(1)[0][0]
            stable_emotion = EMOTIONS[stable_id]
            stable_color   = EMOTION_COLORS[stable_emotion]
            stable_conf    = np.mean(prob_buffer, axis=0)[stable_id]

        # ── Draw Face Box ─────────────────────────────────────
        cv2.rectangle(display,
                      (x, y), (x+w, y+h),
                      stable_color, 2)

        # ── Emotion Label ─────────────────────────────────────
        emoji = EMOJIS[EMOTIONS.index(stable_emotion)]
        label = f"{stable_emotion}  {stable_conf*100:.0f}%"

        # Label background box
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        ly = max(y - 12, th + 12)
        cv2.rectangle(display,
                      (x, ly - th - 10),
                      (x + tw + 12, ly + 6),
                      stable_color, -1)

        # Label text
        cv2.putText(display, label,
                    (x + 6, ly),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8, (255, 255, 255), 1)

    # ── FPS ───────────────────────────────────────────────────
    if frame_count % 30 == 0:
        fps      = 30 / (time.time() - fps_time)
        fps_time = time.time()

    cv2.putText(display, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # ── Bottom bar ────────────────────────────────────────────
    cv2.putText(display,
                "Facial Emotion Recognition  |  Press Q to quit",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1)

    cv2.imshow("Facial Emotion Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")