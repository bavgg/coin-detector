from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(source="data/images/iamge.jpg", conf=0)
print(results)