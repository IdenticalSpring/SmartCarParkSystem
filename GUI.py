from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient
import math
import time

app = Flask(__name__)

VIDEO_FOLDER = r'D:\sale\AI\Carpark\video'
current_video = None
current_frame = 0
cap = None
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="o1fIdDJs9P0vHNalGCXt"
)

car_count = 0
free_count = 0
log_entries = []

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

car_tracker = Tracker()
free_tracker = Tracker()

def set_video(video_path):
    global cap, current_frame
    if cap:
        cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        cap = None
    else:
        print(f"Video {video_path} opened successfully")
    current_frame = 0

def call_inference_api(image):
    try:
        result = CLIENT.infer(image, model_id="parking-stall-detection/3")

        return result
    except Exception as err:
        print(f"HTTP error occurred: {err}")
        return None

def generate_frames(skip_frames=2):
    global cap, current_frame, car_count, free_count, log_entries
    while cap and cap.isOpened():
        start_time = time.time()

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += skip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = call_inference_api(pil_image)

        if result is None:
            print("Inference API returned an error.")
            continue

        car_rects, free_rects = convert_predictions_to_rects(result['predictions'])

        car_trackers = car_tracker.update(car_rects)
        free_trackers = free_tracker.update(free_rects)

        car_count = len(car_trackers)
        free_count = len(free_trackers)

        annotated_frame = draw_boxes_and_ids(frame, car_trackers, free_trackers)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps}")

        current_frame += 1

def convert_predictions_to_rects(predictions):
    car_rects = []
    free_rects = []
    for prediction in predictions:
        x1 = prediction['x'] - prediction['width'] / 2
        y1 = prediction['y'] - prediction['height'] / 2
        width = prediction['width']
        height = prediction['height']
        rect = [x1, y1, width, height]
        if prediction['class'] == 'car':
            car_rects.append(rect)
        elif prediction['class'] == 'free':
            free_rects.append(rect)
    return car_rects, free_rects

def draw_boxes_and_ids(frame, car_trackers, free_trackers):
    global log_entries
    frame_height, frame_width = frame.shape[:2]
    line_box_width = 10
    horizontal_positions = [0.30, 0.32, 0.60, 0.62, 0.70, 0.72, 0.80, 0.82]
    line_positions = [
        (int(frame_width * pos) - line_box_width // 2, frame_height - 100, line_box_width, 100)
        for pos in horizontal_positions
    ]

    def is_intersecting(car_box, line_boxes):
        x, y, w, h, _ = car_box
        car_rect = [int(x), int(y), int(x + w), int(y + h)]
        for lx, ly, lw, lh in line_boxes:
            line_rect = [lx, ly, lx + lw, ly + lh]
            if (car_rect[0] < line_rect[2] and car_rect[2] > line_rect[0] and
                car_rect[1] < line_rect[3] and car_rect[3] > line_rect[1]):
                return True
        return False

    used_free_centers = set()

    for track in car_trackers:
        x, y, w, h, obj_id = track
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        label = f"ID {obj_id} (car)"

        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        if is_intersecting(track, line_positions):
            car_center = (int(x1 + w / 2), int(y1 + h / 2))

            min_distance = float('inf')
            nearest_free_center = None
            nearest_free_id = None
            for free_track in free_trackers:
                free_x, free_y, free_w, free_h, free_id = free_track
                free_center = (int(free_x + free_w / 2), int(free_y + free_h / 2))

                if free_center in used_free_centers:
                    continue

                distance = math.hypot(free_center[0] - car_center[0], free_center[1] - car_center[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_free_center = free_center
                    nearest_free_id = free_id

            if nearest_free_center:
                used_free_centers.add(nearest_free_center)
                cv2.line(frame, car_center, (car_center[0], nearest_free_center[1]), (255, 0, 0), 1)
                cv2.line(frame, (car_center[0], nearest_free_center[1]), nearest_free_center, (255, 0, 0), 1)

                # Add log entry
                log_entry = f"Car ID {obj_id} go to Slot ID {nearest_free_id}"
                log_entries.append(log_entry)

    for track in free_trackers:
        x, y, w, h, obj_id = track
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        label = f"ID {obj_id} (free)"

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    for lx, ly, lw, lh in line_positions:
        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 1)

    return frame

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_video, current_frame
    if request.method == 'POST':
        if 'video' in request.form:
            video = request.form['video']
            video_path = os.path.join(VIDEO_FOLDER, video)
            set_video(video_path)
            current_video = video
        elif 'action' in request.form:
            action = request.form['action']
            if cap and cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if action == 'forward':
                    current_frame = min(current_frame + 300, frame_count - 1)
                elif action == 'backward':
                    current_frame = max(current_frame - 300, 0)
        return jsonify(success=True)

    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    return render_template('index.html', videos=videos, current_video=current_video)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_info')
def count_info():
    global car_count, free_count
    return jsonify(car_count=car_count, free_count=free_count)

@app.route('/log_info')
def log_info():
    global log_entries
    return jsonify(log_entries=log_entries[-20:])

if __name__ == '__main__':
    app.run(debug=True)
