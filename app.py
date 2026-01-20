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

# Create the file with headers if it doesn't exist yet
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
        # Optimization: Only log to file once every few seconds to avoid spamming
        self.last_log_time = datetime.datetime.now()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every 30 frames (approx 1 second) to keep video smooth
        if self.frame_count % 30 == 0:
            try:
                # --- ACCURACY SETTING ---
                # detector_backend='opencv' is fast. 
                # If you have a powerful PC, change 'opencv' to 'ssd' or 'retinaface' for higher accuracy.
                objs = DeepFace.analyze(
                    img_path = img, 
                    actions = ['gender'],
                    detector_backend = 'opencv', 
                    enforce_detection = False,
                    silent = True
                )
                
                if len(objs) > 0:
                    result = objs[0]
                    self.current_gender = result['dominant_gender']
                    # Confidence score (sometimes DeepFace returns this differently, handling safely)
                    probs = result['gender']
                    self.confidence = probs[self.current_gender]

                    # --- LOGGING TO FILE ---
                    # Only log if confidence is high (> 60%) to avoid errors
                    if self.confidence > 60:
                        now = datetime.datetime.now()
                        # Limit logging to once every 2 seconds per person
                        if (now - self.last_log_time).total_seconds() > 2.0:
                            with open(LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([now, self.current_gender, f"{self.confidence:.2f}%"])
                            self.last_log_time = now

            except Exception as e:
                print(f"Error: {e}")

        self.frame_count += 1

        # --- DRAWING ON SCREEN ---
        # 1. Background Box (Black)
        cv2.rectangle(img, (20, 20), (450, 100), (0,0,0), -1)
        
        # 2. Text (Gender)
        color = (0, 255, 0) if self.current_gender == "Man" else (255, 0, 255) # Green for Man, Magenta for Woman
        cv2.putText(img, f"Gender: {self.current_gender}", (30, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 3. Text (Confidence)
        cv2.putText(img, f"Conf: {self.confidence:.1f}%", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE WEB APP UI ---
st.title("Gender Recognition AI")
st.write("Real-time gender detection system. Data is automatically saved to CSV.")

# Create a download button for the log file
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