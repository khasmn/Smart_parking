from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

car_cls_id = 2
parking_lot = 12

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    result = model.predict(source=frame, conf=0.25, save = False)

    occupied_slot = 0

    for result in result[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.cpu().numpy()
        if cls == car_cls_id:
            occupied_slot += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            lable = f'Car: {conf:.2f}'
            cv2.putText(frame, lable, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        status_text = f"Parked: {occupied_slot}/{parking_lot}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Parking Lot detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
