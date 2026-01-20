import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from deepface import DeepFace
import datetime
import csv
import os

# --- 1. SETUP LOGGING ---
LOG_FILE = 'gender_log.csv'

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Gender', 'Confidence'])

# --- 2. THE PROCESSOR ---
class GenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_gender = "Initializing..."
        self.confidence = 0.0
        self.last_log_time = datetime.datetime.now()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every 30 frames (approx 1 second)
        if self.frame_count % 30 == 0:
            try:
                objs = DeepFace.analyze(
                    img_path = img, 
                    actions = ['gender'],
                    detector_backend = 'opencv', 
                    enforce_detection = False,
                    silent = True
                )
                
                if len(objs) > 0:
                    result = objs[0]
                    raw_gender = result['dominant_gender']
                    
                    # --- TRANSLATION STEP ---
                    # Convert "Man" -> "Male" and "Woman" -> "Female"
                    if raw_gender == "Man":
                        self.current_gender = "Male"
                    elif raw_gender == "Woman":
                        self.current_gender = "Female"
                    else:
                        self.current_gender = raw_gender # Fallback

                    # Get confidence
                    probs = result['gender']
                    # We need to look up the confidence using the original raw key ("Man" or "Woman")
                    self.confidence = probs[raw_gender]

                    # --- LOGGING ---
                    if self.confidence > 60:
                        now = datetime.datetime.now()
                        if (now - self.last_log_time).total_seconds() > 2.0:
                            with open(LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([now, self.current_gender, f"{self.confidence:.2f}%"])
                            self.last_log_time = now

            except Exception as e:
                print(f"Error: {e}")

        self.frame_count += 1

        # --- DRAWING ON SCREEN ---
        cv2.rectangle(img, (20, 20), (450, 100), (0,0,0), -1)
        
        # Color logic: Green for Male, Magenta for Female
        color = (0, 255, 0) if self.current_gender == "Male" else (255, 0, 255)
        
        cv2.putText(img, f"Gender: {self.current_gender}", (30, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(img, f"Conf: {self.confidence:.1f}%", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE WEB APP UI ---
st.title("Gender Recognition AI")
st.write("Real-time gender detection system. Data is automatically saved to CSV.")

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as file:
        st.download_button(
            label="Download Excel/CSV Logs",
            data=file,
            file_name="gender_logs.csv",
            mime="text/csv"
        )

webrtc_streamer(
    key="gender-detect",
    video_processor_factory=GenderProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
