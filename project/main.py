from ultralytics import YOLO
import cv2

# Load the YOLO trained model
model = YOLO('car_parking.pt')

# Print all class names in the model
print("Class names in the model:", model.names)

# Define the class ID
vacant_class_id = 1
occupied_class_id = 0

# Open the camera
cap = cv2.VideoCapture(0) # 0 for the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# loop through the frames
while True:
    # Capture a frame from the cam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Use AI model to detect parking slots
    results = model.predict(source=frame, conf=0.7, save=False)

    # Initialize counters for parking slots and cars before detecting
    vacant_count = 0
    occupied_count = 0

    # Draw bounding boxes for detected slots
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.cpu().numpy()
        if int(cls) == vacant_class_id:
            vacant_count += 1  # Increment the counter
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

    # Resize the frame to fit the screen
    resized_frame = cv2.resize(frame, (800, 600))

    # Display the frame with detections
    cv2.imshow("Parking Slot Detection", resized_frame)

    # Wait for a key press and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()