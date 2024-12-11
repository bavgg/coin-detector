from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(source="data/images/test/iloveyou.1a9307e2-b436-11ef-bec6-d68e14d33a98.jpg", conf=0)
print(results)