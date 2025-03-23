from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolo11n.pt')

# Define the class ID for "car" (update this based on your model's class mapping)
car_cls_id = 2
parking_lot = 12

# Load the image from your file
image_path = "e7a7e9c9-IMG_1625.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Perform inference on the image
results = model.predict(source=frame, conf=0.25, save=False)

# Initialize the counter for occupied parking slots
occupied_slot = 0

# Filter detections for "car" and draw bounding boxes
for result in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = result.cpu().numpy()
    if int(cls) == car_cls_id:
        occupied_slot += 1

        # Draw a rectangle around the car
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Car: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the parking slot status
status_text = f"Parked: {occupied_slot}/{parking_lot}"
cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("Parking Lot Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()