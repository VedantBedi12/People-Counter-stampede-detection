from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import time
from collections import deque
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolov5su.pt')

CAMERAS = {
    "Cam1": 0,
    "Cam2": "videos/sample4.mp4",
    "Cam3": "videos/sample2.mp4",
    "Cam4": "videos/sample3.mp4",
    "EntryCam": "videos/sample2.mp4"
}

thresholds = {
    "Cam1": 20,
    "Cam2": 34,
    "Cam3": 7,
    "Cam4": 10,
    "EntryCam": 1000  # Irrelevant for entry tracking
}

alert_sent = {cam: False for cam in CAMERAS if cam != "EntryCam"}
alerts = {cam: "" for cam in CAMERAS if cam != "EntryCam"}
crowd_history = {cam: deque(maxlen=600) for cam in CAMERAS if cam != "EntryCam"}

entry_count = 0
prev_centers = []


def generate_frames(camera_name):
    global entry_count, prev_centers

    cap = cv2.VideoCapture(CAMERAS[camera_name])
    skip_interval = 5
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_interval != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)
        detections = results[0].boxes.data

        count = 0
        current_centers = []

        for det in detections:
            cls_id = int(det[5])
            if cls_id == 0:
                count += 1
                x1, y1, x2, y2 = map(int, det[:4])
                center_y = (y1 + y2) // 2
                current_centers.append(center_y)
                color = (0, 0, 255) if count > thresholds.get(camera_name, 1000) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if camera_name != "EntryCam":
            timestamp = time.time()
            crowd_history[camera_name].append((timestamp, count))

            if count > thresholds[camera_name]:
                cv2.putText(frame, "\u26a0\ufe0f Large Crowd Detected!", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not alert_sent[camera_name]:
                    alerts[camera_name] = f"\u26a0\ufe0f Crowd Alert on {camera_name}: {count} people detected!"
                    alert_sent[camera_name] = True
            else:
                alerts[camera_name] = ""
                alert_sent[camera_name] = False

            cv2.putText(frame, f'People: {count}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # Entry logic only for EntryCam
            line_y = 300
            cv2.line(frame, (0, line_y), (640, line_y), (255, 255, 0), 2)

            for prev, curr in zip(prev_centers, current_centers):
                if prev < line_y and curr >= line_y:
                    entry_count += 1

            prev_centers = current_centers

            cv2.putText(frame, f'Entry Count: {entry_count}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for cam in thresholds:
            if cam in request.form:
                thresholds[cam] = int(request.form[cam])
        return redirect(url_for('index'))
    return render_template('index.html', cameras=CAMERAS, thresholds=thresholds, alerts=alerts)


@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    return Response(generate_frames(camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/graph/<camera_name>')
def graph(camera_name):
    data = list(crowd_history[camera_name])
    times = [time.strftime('%H:%M:%S', time.localtime(t)) for t, _ in data]
    counts = [c for _, c in data]
    return render_template('graph.html', times=times, counts=counts, cam=camera_name)


@app.route('/entry_count')
def entry():
    return jsonify(count=entry_count)


if __name__ == '__main__':
    app.run(debug=True)