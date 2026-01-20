import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import datetime
import csv
import os
from PIL import Image

# --- 1. SETUP LOGGING ---
LOG_FILE = 'gender_log.csv'

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Gender', 'Confidence'])

# --- 2. THE WEB INTERFACE ---
st.title("Facial Gender Detection System")
st.write("1. Allow camera access.\n2. Click 'Take Photo' to detect gender.")

# --- 3. THE CAMERA INPUT (Standard HTML5 Camera) ---
# This uses standard web upload, so it works on ALL mobile networks automatically.
img_file_buffer = st.camera_input("Take a Selfie")

if img_file_buffer is not None:
    # Convert the file buffer to an image that OpenCV can read
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.write("Analyzing... Please wait.")
    
    try:
        # --- 4. HIGH ACCURACY DETECTION ---
        # We can afford to use 'RetinaFace' because we are only processing ONE image,
        # so speed doesn't matter as much as accuracy.
        objs = DeepFace.analyze(
            img_path = cv2_img, 
            actions = ['gender'],
            detector_backend = 'retinaface', # Maximum Accuracy
            enforce_detection = False,
            silent = True
        )

        if len(objs) > 0:
            result = objs[0]
            
            # --- GENDER LOGIC ---
            gender_probs = result['gender']
            woman_score = gender_probs['Woman']
            man_score = gender_probs['Man']

            # Sensitivity Tweak
            if woman_score > 40: 
                gender = "Female"
                confidence = woman_score
            else:
                gender = "Male"
                confidence = man_score

            # --- DISPLAY RESULT ---
            # Create two columns for a nice layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Show the gender in big text
                if gender == "Male":
                    st.success(f"## Detected: {gender}")
                else:
                    st.error(f"## Detected: {gender}") # Red/Pink color for female
            
            with col2:
                st.metric(label="Confidence", value=f"{confidence:.1f}%")

            # --- LOG TO FILE ---
            if confidence > 50:
                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.datetime.now(), gender, f"{confidence:.2f}%"])
                st.toast("Data saved to CSV!", icon="✅")

        else:
            st.warning("No face detected. Please hold the camera closer.")

    except Exception as e:
        st.error(f"Error during analysis: {e}")

# --- 5. DOWNLOAD LOGS ---
st.markdown("---")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as file:
        st.download_button(
            label="Download Logs",
            data=file,
            file_name="gender_logs.csv",
            mime="text/csv"
        )

