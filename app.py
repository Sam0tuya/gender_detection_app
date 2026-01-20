import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from twilio.rest import Client
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

# --- 2. THE PROCESSOR CLASS (The AI Brain) ---
class GenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_gender = "Initializing..."
        self.confidence = 0.0
        self.last_log_time = datetime.datetime.now()

    def recv(self, frame):
        # Convert video frame to a format Python understands (NumPy array)
        img = frame.to_ndarray(format="bgr24")
        
        # --- OPTIMIZATION ---
        # RetinaFace is accurate but heavy. We only analyze once every 30 frames (approx 1 sec)
        # to prevent the video from lagging.
        if self.frame_count % 30 == 0:
            try:
                # Run the AI
                objs = DeepFace.analyze(
                    img_path = img, 
                    actions = ['gender'],
                    detector_backend = 'retinaface', # High Accuracy Mode
                    enforce_detection = False,
                    silent = True
                )
                
                if len(objs) > 0:
                    result = objs[0]
                    
                    # --- GENDER LOGIC & BIAS FIX ---
                    gender_probs = result['gender']
                    woman_score = gender_probs['Woman']
                    man_score = gender_probs['Man']

                    # "Sensitivity Tweak": If AI is > 40% sure it's a woman, trust it.
                    # This helps detect females better in poor lighting.
                    if woman_score > 40: 
                        raw_gender = "Woman"
                        self.confidence = woman_score
                    else:
                        raw_gender = "Man"
                        self.confidence = man_score

                    # --- TRANSLATION (Man/Woman -> Male/Female) ---
                    if raw_gender == "Man":
                        self.current_gender = "Male"
                    elif raw_gender == "Woman":
                        self.current_gender = "Female"
                    
                    # --- LOGGING TO FILE ---
                    # Only log if confidence is decent (> 50%)
                    if self.confidence > 50:
                        now = datetime.datetime.now()
                        # Timer: Don't spam the file; wait 2 seconds between logs
                        if (now - self.last_log_time).total_seconds() > 2.0:
                            with open(LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([now, self.current_gender, f"{self.confidence:.2f}%"])
                            self.last_log_time = now

            except Exception as e:
                # If no face is found or an error occurs, just skip this frame
                pass

        self.frame_count += 1

        # --- DRAWING ON SCREEN ---
        # 1. Background Box (Black) for text visibility
        cv2.rectangle(img, (20, 20), (450, 100), (0,0,0), -1)
        
        # 2. Text Color Logic: Green for Male, Magenta for Female
        color = (0, 255, 0) if self.current_gender == "Male" else (255, 0, 255)
        
        # 3. Draw Gender Text
        cv2.putText(img, f"Gender: {self.current_gender}", (30, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 4. Draw Confidence Text
        cv2.putText(img, f"Conf: {self.confidence:.1f}%", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE WEB INTERFACE ---
st.title("Gender Recognition AI (Mobile Ready)")
st.write("Real-time gender detection system. Data is automatically saved to CSV.")

# Download Button Logic
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as file:
        st.download_button(
            label="Download Excel/CSV Logs",
            data=file,
            file_name="gender_logs.csv",
            mime="text/csv"
        )

# --- 4. TWILIO TURN SERVER SETUP (Crucial for Mobile) ---
# REPLACE THE VALUES BELOW WITH YOUR REAL TWILIO KEYS
account_sid = "US5a3f66ce736ed16393fe78d236b28242"  # <--- PASTE YOUR SID HERE
auth_token = "HUS5GEGLEWS4BKG9W296GFBJ"    # <--- PASTE YOUR TOKEN HERE

ice_servers = []

try:
    # Attempt to fetch powerful TURN servers from Twilio
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    ice_servers = token.ice_servers
except Exception as e:
    # Fallback to free Google STUN servers (May fail on mobile networks)
    # st.warning("Twilio connection failed. Using basic servers.")
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- 5. START THE CAMERA ---
webrtc_streamer(
    key="gender-detect",
    video_processor_factory=GenderProcessor,
    rtc_configuration={"iceServers": ice_servers}
)
