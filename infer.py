import cv2
from ultralytics import YOLO

model_path = "/network/scratch/i/islamria/Rrm/exp/waiter_calling/model/best.pt"
model = YOLO(model_path)

# Step 2: Open the video file
video_path = "/network/scratch/i/islamria/Rrm/exp/waiter_calling/raw_data/desk_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Error opening video file")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare the output video writer
output_path = "/network/scratch/i/islamria/Rrm/exp/waiter_calling/output/person_identification.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Step 3: Define the ranges and mapping function
ranges = [
    {"object": "Tanvir", "x_range": (550, 570), "y_range": (450, 485)},
    {"object": "Anik", "x_range": (1130, 1150), "y_range": (450, 465)},
    {"object": "Toufiq", "x_range": (795, 880), "y_range": (450, 461)},
    {"object": "Imran", "x_range": (1135, 1160), "y_range": (470, 520)},
    {"object": "Murfad", "x_range": (955, 995), "y_range": (470, 495)},
    {"object": "Emon", "x_range": (1210, 1230), "y_range": (475, 505)},
    {"object": "Shafayet", "x_range": (700, 720), "y_range": (465, 485)},
    {"object": "Faisal", "x_range": (814, 826), "y_range": (462, 475)},
]

def map_center_to_object(x, y):
    """
    Maps the center coordinates (x, y) to an object name based on predefined ranges.
    """
    for region in ranges:
        if region["x_range"][0] <= x <= region["x_range"][1] and region["y_range"][0] <= y <= region["y_range"][1]:
            return region["object"]
    return None

frame_skip = 2  # Process every 3rd frame
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    object_name = None

    if frame_index % frame_skip == 0:
        results = model.predict(frame, save=False, conf=0.5)

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2 = map(int, box[:4])  # When working with images yolo gives the bbox coordinates.
                                                # Instead of center_x, center_y, bbox_width, bbox_height. the txt file do only.
            width = x2 - x1
            height = y2 - y1

            x_center = x1 + width // 2
            y_center = y1 + height // 2

            print(f"Box: ({x1}, {y1}) -> ({x2}, {y2}) | Center: ({x_center}, {y_center}) | BBox Size: ({width}, {height})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # cv2.circle(frame, (x_center, y_center), 10, (255, 0, 0), -1)  # Blue circle

            # Map center to an object
            object_name = map_center_to_object(x1, y1)
            print(object_name)

        if object_name:
            cv2.putText(frame, object_name, (x2 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (0, 0, 255), 3)  # Red text


    out.write(frame)
    frame_index += 1

cap.release()
out.release()

print(f"Processed video saved to {output_path}")
