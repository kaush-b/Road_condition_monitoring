import streamlit as st
import cv2
import requests
import tempfile
import os
import math
from PIL import Image, ImageDraw

# Roboflow model details
API_KEY = "QxItssNeCicWL3vyn2kQ"
PROJECT_NAME = "road_monitoring-bwx5f"
MODEL_VERSION = "2"

st.title("Pothole Detection App")
st.write("Upload a video to detect potholes.")

# Detection helper
def is_new_pothole(x, y, prev_detections, threshold=50):
    for px, py in prev_detections:
        if math.hypot(px - x, py - y) < threshold:
            return False
    return True

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)
    st.write("Processing video...")

    cap = cv2.VideoCapture(temp_video_path)

    if cap.isOpened():
        frame_rate = 5
        count = 0
        total_potholes = 0
        FRAME_DISPLAY_LIMIT = 10
        stframe = st.empty()
        seen_potholes = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_rate == 0:
                _, img_encoded = cv2.imencode(".jpg", frame)
                img_bytes = img_encoded.tobytes()

                response = requests.post(
                    f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={API_KEY}",
                    files={"file": img_bytes},
                    data={"confidence": "40", "overlap": "30"},
                )

                predictions = response.json().get("predictions", [])
                new_count = 0

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(image_pil)

                for pred in predictions:
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    if is_new_pothole(x, y, seen_potholes):
                        seen_potholes.append((x, y))
                        total_potholes += 1
                        new_count += 1

                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)

                stframe.image(image_pil, caption=f"New: {new_count} | Total: {total_potholes}", use_column_width=True)

                FRAME_DISPLAY_LIMIT -= 1
                if FRAME_DISPLAY_LIMIT <= 0:
                    break

            count += 1

        cap.release()
        os.remove(temp_video_path)
        st.success(f"Finished! Total potholes detected: {total_potholes}")

