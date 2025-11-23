import cv2
import numpy as np
from pathlib import Path

# ----------------------------
# PATH SETUP (GitHub-Friendly)
# ----------------------------
# Resolve the directory where this script is located
base_dir = Path(__file__).resolve().parent

# Model files (must be in the same folder)
prototxt_path = base_dir / "MobileNetSSD_deploy.prototxt"
model_path = base_dir / "MobileNetSSD_deploy.caffemodel"

# Load the SSD model
net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))

# ---------------------------------------
# Object category labels for MobileNetSSD
# ---------------------------------------
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Random colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ----------------------------
# WEBCAM INITIALIZATION
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Convert to blob for DNN input
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5
    )

    net.setInput(blob)
    detections = net.forward()

    # ----------------------------
    # LOOP THROUGH DETECTIONS
    # ----------------------------
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Bounding box + label
            label = f"{CLASSES[idx]}: {confidence*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[idx], 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLORS[idx], 2
            )

    # Display the result
    cv2.imshow("Webcam Object Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
