from flask import Flask, render_template, request, redirect, Response, jsonify
from ultralytics import YOLO
import cv2
import os
import time
import base64
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DETECTION_FOLDER = "static/detections"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

model = YOLO("model/best.pt")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print("🚀 Scanning video for fire...")

        cap = cv2.VideoCapture(filepath)
        fire_detected = False
        detected_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, device="cpu", conf=0.4, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    fire_detected = True
                    detected_frame = r.plot()
                    print(f"🔥 Fire detected! Confidence: {float(r.boxes.conf[0]):.2f}")
                    break

            if fire_detected:
                break

        cap.release()

        if fire_detected:
            save_path = os.path.join(DETECTION_FOLDER, "fire_detected_frame.jpg")
            cv2.imwrite(save_path, detected_frame)
            return render_template('detect.html', detected=True, frame_path=save_path)
        else:
            return render_template('detect.html', detected=False)

    return render_template('detect.html', detected=None)

@app.route('/live')
def live():
    return render_template('live.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(source=frame, device="cpu", conf=0.4, verbose=False)
            for r in results:
                frame = r.plot()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/safety')
def safety():
    return render_template('safety.html')

@app.route('/station')
def station():
    return render_template('station.html')

# ======================== 🔥 NEW: Verify Snapshot for Live Detection ======================== #
@app.route('/verify_snapshot', methods=['POST'])
def verify_snapshot():
    data = request.get_json()
    image_data = data.get('image')
    if not image_data:
        return jsonify({"error": "No image received"}), 400

    image_bytes = base64.b64decode(image_data.split(",")[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(source=frame, device="cpu", conf=0.4, verbose=False)
    fire_detected = False
    detected_frame = None

    for r in results:
        if len(r.boxes) > 0:
            fire_detected = True
            detected_frame = r.plot()
            print(f"🔥 Fire detected in live snapshot! Confidence: {float(r.boxes.conf[0]):.2f}")
            break

    if fire_detected:
        os.makedirs(DETECTION_FOLDER, exist_ok=True)
        save_path = os.path.join(DETECTION_FOLDER, f"fire_live_{int(time.time())}.jpg")
        cv2.imwrite(save_path, detected_frame)
        return jsonify({"fire": True, "frame_path": save_path})
    else:
        return jsonify({"fire": False})
# =========================================================================================== #

if __name__ == '__main__':
    app.run(debug=True)
