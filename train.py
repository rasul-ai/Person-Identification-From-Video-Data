from ultralytics import YOLO

dataset_path = './custom.yaml'

model = YOLO('yolov8n.pt')

model.train(
    data=dataset_path,
    epochs=50,
    imgsz=768,
    batch=8,
    name='yolov8_training',
    device=0
)


results = model.val()