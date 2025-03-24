from ultralytics import YOLO

model = YOLO('car_parking.pt')

print("Class names in the model:", model.names)