import cv2
from ultralytics import YOLO


# ESP32-CAM video stream URL
# Replace this URL with the actual IP address of your ESP32-CAM stream.
# Example for ESP32-CAM CameraWebServer:
# camera_url = "http://192.168.123.16:81/stream"

camera_url = "http://192.168.123.7:8080/stream.mjpg"


# Load YOLOv8 lightweight object detection model
model = YOLO("yolov8n.pt")


# Open video stream from ESP32-CAM
cap = cv2.VideoCapture(camera_url)


if not cap.isOpened():
    print("Camera stream not opened. Check ESP32-CAM URL and Wi-Fi connection.")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to get frame from ESP32-CAM stream.")
        break

    # Run YOLO object detection on the current frame
    results = model(frame, stream=True)

    object_detected = False

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Confidence threshold
            if confidence > 0.45:
                object_detected = True

                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box around detected object
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Add object name and confidence value
                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                print(f"Detected object: {class_name}, confidence: {confidence:.2f}")

    # Display robot decision based on object detection
    if object_detected:
        cv2.putText(
            frame,
            "OBJECT DETECTED - STOP ROBOT",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            frame,
            "NO OBJECT - MOVE FORWARD",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    # Show video stream with YOLO detection result
    cv2.imshow("ESP32-CAM YOLO Object Detection", frame)

    # Press q to stop the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
