# ============================================================
#  STREAMLIT FRONTEND — FACIAL EMOTION RECOGNITION
#  Real-time webcam emotion detection
# ============================================================
#
#  SETUP:
#  pip install streamlit streamlit-webrtc ultralytics tensorflow opencv-python av
#
#  RUN:
#  streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
import time
import collections
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense — Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #312e81;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.3s;
    }

    .metric-card:hover { border-color: #7c3aed; }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1e1b4b;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSidebar"] {
        background: #0d0d1a;
        border-right: 1px solid #1e1b4b;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover { opacity: 0.85; }

    .bar-wrap {
        background: #1e1b4b22;
        border-radius: 6px;
        overflow: hidden;
        height: 10px;
        margin: 3px 0 8px 0;
    }

    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_CONFIG = {
    "Angry"   : {"color": "#ef4444", "emoji": "😡"},
    "Disgust" : {"color": "#8b5cf6", "emoji": "🤢"},
    "Fear"    : {"color": "#f59e0b", "emoji": "😨"},
    "Happy"   : {"color": "#10b981", "emoji": "😄"},
    "Sad"     : {"color": "#3b82f6", "emoji": "😢"},
    "Surprise": {"color": "#f97316", "emoji": "😲"},
    "Neutral" : {"color": "#6b7280", "emoji": "😐"},
}

# ── Session State ─────────────────────────────────────────────
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = collections.deque(maxlen=50)
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "dominant_emotion" not in st.session_state:
    st.session_state.dominant_emotion = "—"

# ── Load Models (cached) ──────────────────────────────────────
@st.cache_resource
def load_models():
    emotion_model = tf.keras.models.load_model(r"C:\Users\RIA MALHOTRA\OneDrive\Desktop\New folder\FacialEmotionRecognition\models\best_emotion_model.keras")
    yolo_model    = YOLO("yolov8n.pt")          # Use yolov8n-face.pt for face-only
    return emotion_model, yolo_model

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">⚙ Configuration</p>', unsafe_allow_html=True)

    model_path = st.text_input("Model Path", value = r"C:\Users\RIA MALHOTRA\OneDrive\Desktop\New folder\FacialEmotionRecognition\models\best_emotion_model.keras")
    conf_thresh = st.slider("YOLO Confidence", 0.1, 0.9, 0.45, 0.05)
    show_bars   = st.checkbox("Show probability bars", value=True)
    show_fps    = st.checkbox("Show FPS", value=True)

    st.markdown('<p class="section-title">📊 Session Stats</p>', unsafe_allow_html=True)

    stats_placeholder = st.empty()

    st.markdown('<p class="section-title">🎭 Emotion Legend</p>', unsafe_allow_html=True)
    for em, cfg in EMOTION_CONFIG.items():
        st.markdown(
            f'<span style="color:{cfg["color"]};font-size:1.1rem">{cfg["emoji"]} </span>'
            f'<span style="color:#e2e8f0">{em}</span>',
            unsafe_allow_html=True
        )

# ── Main Header ───────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">EmotiSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time facial emotion recognition powered by deep learning & YOLOv8</p>',
                unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("🟢 **Live**", unsafe_allow_html=True)

st.markdown("---")

# ── Metric Cards ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

metric_placeholder = {
    "detections" : m1.empty(),
    "dominant"   : m2.empty(),
    "confidence" : m3.empty(),
    "uptime"     : m4.empty(),
}

def render_metrics(total, dominant, conf, uptime_s):
    for key, ph in metric_placeholder.items():
        with ph.container():
            if key == "detections":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Total Detections</div>
                </div>""", unsafe_allow_html=True)
            elif key == "dominant":
                em = dominant if dominant in EMOTION_CONFIG else "—"
                color = EMOTION_CONFIG[em]["color"] if em in EMOTION_CONFIG else "#64748b"
                emoji = EMOTION_CONFIG[em]["emoji"] if em in EMOTION_CONFIG else "🎭"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{emoji} {em}</div>
                    <div class="metric-label">Dominant Emotion</div>
                </div>""", unsafe_allow_html=True)
            elif key == "confidence":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{conf:.0%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>""", unsafe_allow_html=True)
            elif key == "uptime":
                m, s = divmod(int(uptime_s), 60)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m:02d}:{s:02d}</div>
                    <div class="metric-label">Session Uptime</div>
                </div>""", unsafe_allow_html=True)

render_metrics(0, "—", 0.0, 0)

st.markdown("---")

# ── Video Processor ──────────────────────────────────────────
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion_model, self.yolo = load_models()
        self.input_shape = self.emotion_model.input_shape[1:3]
        self.conf_thresh = conf_thresh
        self.show_bars   = show_bars
        self.last_probs  = [0.0] * 7
        self.last_emotion = "Neutral"
        self.frame_times = collections.deque(maxlen=30)
        self.start_time  = time.time()

    def preprocess_face(self, face_img):
        face_gray    = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (self.input_shape[1], self.input_shape[0]))
        if self.input_shape == (48, 48):
            return face_resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
        face_rgb = np.stack([face_resized] * 3, axis=-1)
        face_rgb = cv2.resize(face_rgb, (self.input_shape[1], self.input_shape[0]))
        return face_rgb.reshape(1, *self.input_shape, 3).astype(np.float32) / 255.0

    def draw_bars(self, frame, probs, x1, y1, w):
        bx = x1 + w + 12
        by = y1
        bar_max = 110
        if bx + bar_max + 65 > frame.shape[1]:
            bx = max(0, x1 - bar_max - 75)
        for i, (em, p) in enumerate(zip(EMOTIONS, probs)):
            cfg   = EMOTION_CONFIG[em]
            color = tuple(int(cfg["color"][j:j+2], 16) for j in (5, 3, 1))  # BGR
            row_y = by + i * 18
            # Background
            cv2.rectangle(frame, (bx, row_y), (bx + bar_max, row_y + 12), (30, 30, 40), -1)
            # Fill
            fill = int(p * bar_max)
            if fill > 0:
                cv2.rectangle(frame, (bx, row_y), (bx + fill, row_y + 12), color, -1)
            # Label
            cv2.putText(frame, f"{em[:3]} {p*100:.0f}%",
                        (bx + bar_max + 5, row_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 210), 1)

    def recv(self, frame):
        t0  = time.time()
        img = frame.to_ndarray(format="bgr24")
        out = img.copy()

        results = self.yolo(img, conf=self.conf_thresh, classes=[0], verbose=False)
        boxes   = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            face   = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            inp   = self.preprocess_face(face)
            probs = self.emotion_model.predict(inp, verbose=0)[0]
            eid   = np.argmax(probs)
            em    = EMOTIONS[eid]
            conf  = probs[eid]

            cfg   = EMOTION_CONFIG[em]
            color_hex = cfg["color"]
            color_bgr = tuple(int(color_hex[j:j+2], 16) for j in (5, 3, 1))

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
            # Label
            label = f"{cfg['emoji']} {em}  {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            ly = max(y1 - 10, th + 10)
            cv2.rectangle(out, (x1, ly - th - 8), (x1 + tw + 10, ly + 5), color_bgr, -1)
            cv2.putText(out, label, (x1 + 5, ly),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

            if self.show_bars:
                self.draw_bars(out, probs, x1, y1, x2 - x1)

            # Update session state
            self.last_probs  = probs.tolist()
            self.last_emotion = em
            st.session_state.emotion_history.append(em)
            st.session_state.total_detections += 1
            st.session_state.dominant_emotion = (
                collections.Counter(st.session_state.emotion_history).most_common(1)[0][0]
            )

        # FPS
        self.frame_times.append(time.time() - t0)
        if show_fps and self.frame_times:
            fps = 1 / (sum(self.frame_times) / len(self.frame_times))
            cv2.putText(out, f"FPS {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80), 2)

        # Watermark
        cv2.putText(out, "EmotiSense | YOLOv8 + Deep Learning",
                    (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 180), 1)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ── Webcam Stream ─────────────────────────────────────────────
st.markdown('<p class="section-title">📷 Live Camera Feed</p>', unsafe_allow_html=True)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ── Live Emotion Probabilities ────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">📊 Live Emotion Probabilities</p>',
            unsafe_allow_html=True)

prob_placeholder = st.empty()

if ctx.state.playing and ctx.video_processor:
    start_time = time.time()
    while ctx.state.playing:
        probs  = ctx.video_processor.last_probs
        dom    = ctx.video_processor.last_emotion
        uptime = time.time() - start_time
        total  = st.session_state.total_detections
        conf   = max(probs) if probs else 0.0

        render_metrics(total, dom, conf, uptime)

        # Probability bars
        with prob_placeholder.container():
            for em, p in zip(EMOTIONS, probs):
                cfg   = EMOTION_CONFIG[em]
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown(f"**{cfg['emoji']} {em}**")
                with col2:
                    st.markdown(
                        f"""<div class="bar-wrap">
                            <div class="bar-fill" style="width:{p*100:.1f}%;background:{cfg['color']}"></div>
                        </div>
                        <span style="color:#94a3b8;font-size:0.82rem">{p*100:.1f}%</span>""",
                        unsafe_allow_html=True
                    )

        # Sidebar stats
        if st.session_state.emotion_history:
            counts = collections.Counter(st.session_state.emotion_history)
            with stats_placeholder.container():
                for em in EMOTIONS:
                    pct = counts.get(em, 0) / len(st.session_state.emotion_history) * 100
                    st.markdown(
                        f'<span style="color:{EMOTION_CONFIG[em]["color"]}">'
                        f'{EMOTION_CONFIG[em]["emoji"]} {em}</span> — '
                        f'<span style="color:#94a3b8">{pct:.1f}%</span>',
                        unsafe_allow_html=True
                    )

        time.sleep(0.1)

else:
    st.info("👆 Click **START** above to begin real-time emotion detection.")
    # Show placeholder bars
    for em in EMOTIONS:
        cfg = EMOTION_CONFIG[em]
        col1, col2 = st.columns([1, 9])
        with col1:
            st.markdown(f"**{cfg['emoji']} {em}**")
        with col2:
            st.markdown(
                f"""<div class="bar-wrap">
                    <div class="bar-fill" style="width:0%;background:{cfg['color']}"></div>
                </div>
                <span style="color:#94a3b8;font-size:0.82rem">0.0%</span>""",
                unsafe_allow_html=True
            )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:0.8rem">'
    'EmotiSense · Built with YOLOv8 + TensorFlow · FER2013 Dataset'
    '</p>',
    unsafe_allow_html=True
)