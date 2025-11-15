from ultralytics import YOLO
import cv2
import os

# Load trained YOLO model
model = YOLO("model/best.pt")

# Path to video
video_path = "video1.mp4"  # change if needed
cap = cv2.VideoCapture(video_path)

print("🚀 Scanning video for fire on CPU...")

fire_detected = False
detected_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(source=frame, device="cpu", conf=0.4, verbose=False)

    for r in results:
        if len(r.boxes) > 0:  # fire detected
            fire_detected = True
            detected_frame = r.plot()  # draw bounding boxes
            print(f"🔥 Fire detected! Confidence: {float(r.boxes.conf[0]):.2f}")
            break

    if fire_detected:
        break

cap.release()

# Save and open detected frame
if fire_detected and detected_frame is not None:
    os.makedirs("detections", exist_ok=True)
    save_path = os.path.join("detections", "fire_detected_frame.jpg")
    cv2.imwrite(save_path, detected_frame)
    print(f"⚠️ Fire detected — frame saved at {save_path}")

    # Auto open image
    os.startfile(save_path)
else:
    print("✅ No fire detected in video.")
