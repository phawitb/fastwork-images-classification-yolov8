# import YOLO model
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8m-cls.pt') # load a pretrained model (recommended for training)
model = YOLO('runs/classify/train3/weights/best.pt')

# Train the model
model.train(
    data='data',
    epochs=100,              # Number of training epochs
    imgsz=640,                # Image size for training
    batch=4,                  # Batch size
    device="mps"              # Device to use ("mps" for Apple Silicon)
)


#train yolo task=detect mode=train epochs=5000 data=data_custom.yaml model=yolov8n.pt imgsz=640 batch=4 device="mps"
#train yolo task=classify mode=train epochs=5000 data=data_custom.yaml model=yolov8n-cls.pt imgsz=640 batch=4 device="mps"
