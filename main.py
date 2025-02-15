import cv2
import torch
import numpy as np
import threading
from ultralytics import YOLO

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  
else:
    device = 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8m.pt").to(device)
model.fuse()

# Initialize cameras
camera_indexes = [0, 1, 2]  # Adjust based on working cameras
caps = [cv2.VideoCapture(i) for i in camera_indexes]
frames_dict = {i: None for i in camera_indexes}

def process_camera(cam_index, cap):
    print(f"Starting camera {cam_index}...")
    if not cap.isOpened():
        print(f"Error: Camera {cam_index} could not be opened!")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from Camera {cam_index}, skipping...")
            break
        
        # Convert BGR to RGB before sending to GPU
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference on the original frame size
        results = model(frame_rgb, conf=0.8, iou=0.6)  
        for result in results:
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Convert to NumPy array
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, cls_ids):
                    if int(cls) == 0 and conf > 0.8:  # Class 0 = Person, stricter filtering
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frames_dict[cam_index] = frame

# Start processing cameras in parallel using threads
threads = []
for i, cap in enumerate(caps):
    thread = threading.Thread(target=process_camera, args=(i, cap), daemon=True)
    thread.start()
    threads.append(thread)

while True:
    any_frame_available = False
    for cam_index, frame in frames_dict.items():
        if frame is not None:
            any_frame_available = True
            cv2.imshow(f"Camera {cam_index+1}", frame)
    
    if not any_frame_available:
        print("No frames available from any camera...")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

for thread in threads:
    thread.join()


print("All cameras stopped.")
