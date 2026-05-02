# ============================================================
#  REAL-TIME EMOTION DETECTION — VS Code / Jupyter
#  Face Detection : YOLOv8 (ultralytics)
#  Emotion Model  : Your saved best model from Colab
# ============================================================
#
#  SETUP STEPS (run once in terminal):
#  pip install ultralytics opencv-python tensorflow numpy
#
#  USAGE:
#  python realtime_emotion.py
#  Press 'q' to quit

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\RIA MALHOTRA\OneDrive\Desktop\New folder\FacialEmotionRecognition\models\best_emotion_model.keras"   # Path to your downloaded model
YOLO_MODEL   = "yolov8n-face.pt"           # YOLOv8 nano face detection weights
                                            # Auto-downloaded on first run
CONF_THRESH  = 0.45                         # YOLOv8 confidence threshold

EMOTIONS     = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Emotion bar colors (BGR for OpenCV)
EMOTION_COLORS = {
    "Angry"   : (0,   0,   220),
    "Disgust" : (130, 0,   130),
    "Fear"    : (200, 100, 0  ),
    "Happy"   : (0,   200, 200),
    "Sad"     : (200, 80,  0  ),
    "Surprise": (0,   160, 255),
    "Neutral" : (120, 120, 120)
}

# ── Load Models ──────────────────────────────────────────────
print("[INFO] Loading emotion model...")
emotion_model = tf.keras.models.load_model(MODEL_PATH)

# Determine input size from model
input_shape = emotion_model.input_shape[1:3]   # (H, W)
print(f"[INFO] Emotion model input size: {input_shape}")

print("[INFO] Loading YOLOv8 face detector...")
# Using YOLOv8 nano — downloads automatically
# If you want a local weights file: YOLO("path/to/yolov8n-face.pt")
yolo = YOLO("yolov8n.pt")   # General object detection (person/face fallback)
# NOTE: For best face detection, use:
# yolo = YOLO("yolov8n-face.pt")
# Download from: https://github.com/akanametov/yolov8-face/releases

print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Helper: Preprocess Face ───────────────────────────────────
def preprocess_face(face_img, target_size):
    """Convert face crop to model input format."""
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (target_size[1], target_size[0]))

    if target_size == (48, 48):
        # Grayscale model (Custom CNN)
        face_arr = face_resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
    else:
        # RGB model (MobileNetV2, EfficientNetB0, ResNet50)
        face_rgb = np.stack([face_resized] * 3, axis=-1)
        face_arr = cv2.resize(face_rgb, (target_size[1], target_size[0]))
        face_arr = face_arr.reshape(1, *target_size, 3).astype(np.float32) / 255.0

    return face_arr

# ── Helper: Draw Emotion Bar Chart ───────────────────────────
def draw_emotion_bars(frame, probs, x, y, w):
    """Draw mini probability bars next to each face."""
    bar_x  = x + w + 10
    bar_y  = y
    bar_h  = 14
    bar_max_w = 100

    if bar_x + bar_max_w + 60 > frame.shape[1]:
        bar_x = max(0, x - bar_max_w - 70)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        filled = int(prob * bar_max_w)
        color  = EMOTION_COLORS[emotion]

        # Background bar
        cv2.rectangle(frame,
                      (bar_x, bar_y + i * (bar_h + 3)),
                      (bar_x + bar_max_w, bar_y + i * (bar_h + 3) + bar_h),
                      (50, 50, 50), -1)
        # Filled bar
        if filled > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y + i * (bar_h + 3)),
                          (bar_x + filled, bar_y + i * (bar_h + 3) + bar_h),
                          color, -1)
        # Label
        cv2.putText(frame,
                    f"{emotion[:3]} {prob*100:.0f}%",
                    (bar_x + bar_max_w + 5, bar_y + i * (bar_h + 3) + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

# ── Main Loop ─────────────────────────────────────────────────
frame_count = 0
fps_time    = cv2.getTickCount()

print("[INFO] Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_count += 1
    display = frame.copy()

    # ── YOLOv8 Face Detection ──────────────────────────────
    results = yolo(frame, conf=CONF_THRESH, classes=[0], verbose=False)
    # class 0 = person in COCO; use yolov8n-face.pt for face-only detection

    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # ── Preprocess & Predict ───────────────────────────
        face_input = preprocess_face(face_crop, input_shape)
        probs      = emotion_model.predict(face_input, verbose=0)[0]
        emotion_id = np.argmax(probs)
        emotion    = EMOTIONS[emotion_id]
        confidence = probs[emotion_id]
        color      = EMOTION_COLORS[emotion]

        # ── Draw Bounding Box ──────────────────────────────
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # ── Emotion Label ──────────────────────────────────
        label   = f"{emotion}  {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        label_y = max(y1 - 10, th + 10)

        # Background rectangle for label
        cv2.rectangle(display,
                      (x1, label_y - th - 8),
                      (x1 + tw + 10, label_y + 5),
                      color, -1)
        cv2.putText(display, label,
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # ── Probability Bars ───────────────────────────────
        draw_emotion_bars(display, probs, x1, y1, x2 - x1)

    # ── FPS Counter ────────────────────────────────────────
    if frame_count % 30 == 0:
        elapsed = (cv2.getTickCount() - fps_time) / cv2.getTickFrequency()
        fps_time = cv2.getTickCount()
        fps = 30 / elapsed
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── Header ─────────────────────────────────────────────
    cv2.putText(display, "Facial Emotion Recognition  |  Press Q to quit",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Emotion Recognition — YOLOv8", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")