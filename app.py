import streamlit as st
import cv2
import requests
import tempfile
import os
from PIL import Image, ImageDraw
from io import BytesIO

# Roboflow model details
API_KEY = "your-api-key"
PROJECT_NAME = "pothole-detection"
MODEL_VERSION = "1"

st.title("Pothole Detection from Video 🎥")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)
    st.write("🔍 Processing video...")

    cap = cv2.VideoCapture(temp_video_path)
    frame_rate = 5  # Process every 5th frame
    count = 0
    total_potholes = 0

    FRAME_DISPLAY_LIMIT = 10  # Only show 10 processed frames

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            # Convert frame to JPEG in memory
            _, img_encoded = cv2.imencode(".jpg", frame)
            img_bytes = img_encoded.tobytes()

            # Send to Roboflow API
            response = requests.post(
                f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={API_KEY}",
                files={"file": img_bytes},
                data={"confidence": "40", "overlap": "30"},
            )

            predictions = response.json().get("predictions", [])

            # Count potholes
            total_potholes += len(predictions)

            # Draw boxes
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(image_pil)

            for pred in predictions:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2
                draw.rectangle([left, top, right, bottom], outline="red", width=3)

            # Show result
            stframe.image(image_pil, caption=f"Detected {len(predictions)} potholes", use_column_width=True)

            FRAME_DISPLAY_LIMIT -= 1
            if FRAME_DISPLAY_LIMIT <= 0:
                break

        count += 1

    cap.release()
    os.remove(temp_video_path)

    st.success(f"Done! Total potholes detected (in sampled frames): {total_potholes}")
