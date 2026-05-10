import os
import cv2
import av
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

MODEL_PATH = "models/best_emotion_model.keras"

EMOTIONS = {
    0: ("Angry", "😠", "#ef4444"),
    1: ("Disgust", "🤢", "#8b5cf6"),
    2: ("Fear", "😨", "#f59e0b"),
    3: ("Happy", "😄", "#10b981"),
    4: ("Neutral", "😐", "#6b7280"),
    5: ("Sad", "😢", "#3b82f6"),
    6: ("Surprise", "😲", "#f97316"),
}

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    return model, face_detector

model, detector = load_resources()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "emotion" not in st.session_state:
    st.session_state.emotion = "Neutral"

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "probs" not in st.session_state:
    st.session_state.probs = [1/7] * 7

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("😊 Facial Emotion Recognition")
st.caption("Real-time emotion detection using CNN + FER2013")

col1, col2 = st.columns([3, 2])

emotion_box = col2.empty()
prob_box = col2.empty()

# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.buffer = deque(maxlen=10)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            probs = model.predict(face, verbose=0)[0]

            emotion_id = np.argmax(probs)

            self.buffer.append(emotion_id)

            stable_id = Counter(self.buffer).most_common(1)[0][0]

            emotion, emoji, color = EMOTIONS[stable_id]
            confidence = probs[stable_id]

            # Store in session
            st.session_state.emotion = emotion
            st.session_state.confidence = confidence
            st.session_state.probs = probs

            # Draw box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            label = f"{emotion} {confidence*100:.0f}%"

            cv2.putText(
                img,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

with col1:
    webrtc_streamer(
        key="emotion-app",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

# --------------------------------------------------
# LIVE UI UPDATE
# --------------------------------------------------

while True:

    emotion = st.session_state.emotion
    confidence = st.session_state.confidence
    probs = st.session_state.probs

    emotion_box.markdown(f"""
    ## {emotion}

    ### Confidence: {confidence*100:.1f}%
    """)

    prob_text = ""

    for i, prob in enumerate(probs):
        name = EMOTIONS[i][0]
        prob_text += f"{name}: {prob*100:.1f}%  \n"

    prob_box.markdown(prob_text)

    time.sleep(0.2)