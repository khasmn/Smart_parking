from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolo11n.pt')

# Define the class ID for "car"
car_class_id = 2

# Define the total number of parking slots
total_parking_slots = 11

# Define the parking slot regions (manually define the coordinates for each slot)
# Format: [(x1, y1, x2, y2), ...]
parking_slots = [
    (50, 50, 150, 150),   # Slot 1
    (160, 50, 260, 150),  # Slot 2
    (270, 50, 370, 150),  # Slot 3
    (380, 50, 480, 150),  # Slot 4
    (490, 50, 590, 150),  # Slot 5
    (50, 160, 150, 260),  # Slot 6
    (160, 160, 260, 260), # Slot 7
    (270, 160, 370, 260), # Slot 8
    (380, 160, 480, 260), # Slot 9
    (490, 160, 590, 260), # Slot 10
    (600, 160, 700, 260)  # Slot 11
]

# Load the image
image_path = "e7a7e9c9-IMG_1625.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Perform inference on the image
results = model.predict(source=frame, conf=0.25, save=False)

# Extract car detections
cars = []
for result in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = result.cpu().numpy()
    if int(cls) == car_class_id:
        cars.append((int(x1), int(y1), int(x2), int(y2)))

# Check which parking slots are occupied
occupied_slots = 0
for i, slot in enumerate(parking_slots):
    slot_occupied = False
    for car in cars:
        # Check if the car overlaps with the parking slot
        if not (car[2] < slot[0] or car[0] > slot[2] or car[3] < slot[1] or car[1] > slot[3]):
            slot_occupied = True
            break
    if slot_occupied:
        occupied_slots += 1
        # Draw the parking slot as occupied (red)
        cv2.rectangle(frame, (slot[0], slot[1]), (slot[2], slot[3]), (0, 0, 255), 2)
        cv2.putText(frame, f"Slot {i+1}: Occupied", (slot[0], slot[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        # Draw the parking slot as free (green)
        cv2.rectangle(frame, (slot[0], slot[1]), (slot[2], slot[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Slot {i+1}: Free", (slot[0], slot[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the parking slot status
status_text = f"Occupied Slots: {occupied_slots}/{total_parking_slots}"
cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Resize the frame to fit your screen (e.g., 800x600)
resized_frame = cv2.resize(frame, (580, 600))  # Adjust the width and height as needed

# Show the resized image with detections
cv2.imshow("Parking Lot Detection", resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()