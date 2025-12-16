import os
from ultralytics import YOLO

def train(cfg):
    model = YOLO("yolov8n.pt")
    data_yaml = cfg.get("data", "datasets/retail.yaml")
    epochs = cfg.get("epochs", 5)
    model.train(data=data_yaml, epochs=epochs, imgsz=640)
    save_path = os.getenv("MODEL_OUT", "/out")
    os.makedirs(save_path, exist_ok=True)
    model.export(format="onnx", imgsz=640, file=os.path.join(save_path, "yolov8_retail.onnx"))
    print("Model trained and exported to", save_path)

if __name__ == "__main__":
    cfg = {'data': os.getenv('DATA_YAML', 'datasets/retail.yaml'), 'epochs': int(os.getenv('EPOCHS', '5'))}
    train(cfg)
