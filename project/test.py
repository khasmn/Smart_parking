from ultralytics import YOLO
import cv2

# Load the YOLO model trained to detect vacant and occupied parking slots
model = YOLO('car_parking.pt')

# Print the class names to verify the class IDs for "vacant" and "occupied"
print("Class names in the model:", model.names)

# Perform inference on an image
results = model.predict(source='Dataset/Dataset/IMG_1734.JPG', conf=0.7)  # Set a reasonable confidence threshold

# Define the class IDs for "vacant" and "occupied"
vacant_class_id = 1  # Replace with the correct ID for "vacant" after checking model.names
occupied_class_id = 0  # Replace with the correct ID for "occupied" after checking model.names

# Load the image
image_path = 'Dataset/Dataset/IMG_1734.JPG'
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Initialize counters for vacant and occupied slots
vacant_count = 0
occupied_count = 0

# Draw bounding boxes for detected slots
for result in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = result.cpu().numpy()
    print(f"Class ID: {int(cls)}, Confidence: {conf:.2f}")
    if int(cls) == vacant_class_id:
        vacant_count += 1  # Increment the vacant counter
        # Draw the bounding box for vacant slots (green)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Add a label with the confidence score
        label = f"Vacant: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    elif int(cls) == occupied_class_id:
        occupied_count += 1  # Increment the occupied counter
        # Draw the bounding box for occupied slots (red)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # Add a label with the confidence score
        label = f"Occupied: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Print the total counts for vacant and occupied slots
print(f"Number of vacant slots detected: {vacant_count}")
print(f"Number of occupied slots detected: {occupied_count}")

# Resize the image to fit the screen (e.g., 800x600)
resized_frame = cv2.resize(frame, (800, 600))  # Adjust the width and height as needed

# Display the resized image with detections
cv2.imshow("Parking Slot Detection", resized_frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()