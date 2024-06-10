# from flask import Flask, render_template, Response, request, jsonify
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import torch
# import math
# import time
# from ultralytics import YOLO
#
# app = Flask(__name__)
#
# VIDEO_FOLDER = r'D:\sale\AI\Carpark\video'
# current_video = None
# current_frame = 0
# cap = None
#
# # Initialize the YOLO model with your own weights
# model = YOLO('models/best (1).pt')
#
# # Ensure CUDA is available and use it if possible
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# print(f"Using device: {device}")
#
# class Tracker:
#     def __init__(self):
#         self.center_points = {}
#         self.id_count = 0
#
#     def update(self, objects_rect):
#         objects_bbs_ids = []
#         for rect in objects_rect:
#             x, y, w, h = rect
#             cx = (x + x + w) // 2
#             cy = (y + y + h) // 2
#             same_object_detected = False
#             for id, pt in self.center_points.items():
#                 dist = math.hypot(cx - pt[0], cy - pt[1])
#                 if dist < 35:
#                     self.center_points[id] = (cx, cy)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break
#             if not same_object_detected:
#                 self.center_points[self.id_count] = (cx, cy)
#                 objects_bbs_ids.append([x, y, w, h, self.id_count])
#                 self.id_count += 1
#         new_center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}
#         self.center_points = new_center_points.copy()
#         return objects_bbs_ids
#
# car_tracker = Tracker()
# free_tracker = Tracker()
#
# def set_video(video_path):
#     global cap, current_frame
#     if cap:
#         cap.release()
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         cap = None
#     else:
#         print(f"Video {video_path} opened successfully")
#     current_frame = 0
#
# def call_inference(image):
#     try:
#         # Convert image to tensor, normalize, and move to GPU
#         image_tensor = torch.from_numpy(image).to(device).float() / 255.0
#         image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, 3, H, W) format
#         results = model(image_tensor)
#         return results
#     except Exception as err:
#         print(f"Error occurred during inference: {err}")
#         return None
#
# def generate_frames(skip_frames=2, target_resolution=(640, 640)):
#     global cap, current_frame
#     while cap and cap.isOpened():
#         start_time = time.time()
#
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         current_frame += skip_frames
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Resize frame to the target resolution
#         resized_frame = cv2.resize(frame, target_resolution)
#
#         # Perform inference using the model
#         result = call_inference(resized_frame)
#
#         if result is None:
#             print("Inference returned an error.")
#             continue
#
#         # Separate predictions by class
#         car_rects, free_rects = convert_predictions_to_rects(result)
#
#         # Update trackers for each class
#         car_trackers = car_tracker.update(car_rects)
#         free_trackers = free_tracker.update(free_rects)
#
#         # Draw bounding boxes and IDs on the frame
#         annotated_frame = draw_boxes_and_ids(resized_frame, car_trackers, free_trackers)
#
#         _, buffer = cv2.imencode('.jpg', annotated_frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#         end_time = time.time()
#         fps = 1 / (end_time - start_time)
#         print(f"FPS: {fps}")
#
#         current_frame += 1
#
# def convert_predictions_to_rects(results):
#     car_rects = []
#     free_rects = []
#     print(f"Results: {results}")  # Debugging statement to inspect results
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             width = x2 - x1
#             height = y2 - y1
#             conf = box.conf[0].cpu().numpy()
#             cls = box.cls[0].cpu().numpy()
#             rect = [x1, y1, width, height]
#             if cls == 2:  # Assuming class 2 is 'car'
#                 car_rects.append(rect)
#             elif cls == 1:  # Assuming class 1 is 'free'
#                 free_rects.append(rect)
#     print(f"Car rects: {car_rects}, Free rects: {free_rects}")  # Debugging statement to verify rects
#     return car_rects, free_rects
#
# def draw_boxes_and_ids(frame, car_trackers, free_trackers):
#     car_count = 0
#     free_count = 0
#
#     for track in car_trackers:
#         x, y, w, h, obj_id = track
#         x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
#         label = f"ID {obj_id} (car)"
#         color = (0, 0, 255)  # Red
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Thinner line
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # Smaller font
#         car_count += 1
#
#     for track in free_trackers:
#         x, y, w, h, obj_id = track
#         x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
#         label = f"ID {obj_id} (free)"
#         color = (0, 255, 0)  # Green
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Thinner line
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # Smaller font
#         free_count += 1
#
#     # Draw entrance lines
#     frame_height, frame_width = frame.shape[:2]
#     line_positions = [(int(frame_width * 0.78), frame_height), (int(frame_width * 0.88), frame_height)]
#     for position in line_positions:
#         cv2.line(frame, position, (position[0], frame_height - 100), (0, 255, 255), 1)  # Thinner line
#
#     # Draw counts on the frame
#     cv2.putText(frame, f"Car count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.putText(frame, f"Free count: {free_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     return frame
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     global current_video, current_frame
#     if request.method == 'POST':
#         if 'video' in request.form:
#             video = request.form['video']
#             video_path = os.path.join(VIDEO_FOLDER, video)
#             set_video(video_path)
#             current_video = video
#         elif 'action' in request.form:
#             action = request.form['action']
#             if cap and cap.isOpened():
#                 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 if action == 'forward':
#                     current_frame = min(current_frame + 300, frame_count - 1)
#                 elif action == 'backward':
#                     current_frame = max(current_frame - 300, 0)
#         return jsonify(success=True)
#
#     videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
#     return render_template('index.html', videos=videos, current_video=current_video)
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == '__main__':
#     app.run(debug=True)
