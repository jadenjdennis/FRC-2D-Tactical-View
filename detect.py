import cv2
import numpy as np

VIDEO = "assets/match.mp4"
START_TIME = 10

MIN_AREA = 500
MIN_RATIO = 80
LEARNING_RATE = 0.0005

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

ret, frame = cap.read()
if not ret:
    raise SystemExit("Could not read video")

# Select the field
x, y, w, h = cv2.selectROI("Select Field", frame)
cv2.destroyWindow("Select Field")

if w == 0 or h == 0:
    raise SystemExit("No field selected")

# Create MOG2
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=35,
    detectShadows=False
)

# Sliders
cv2.namedWindow("MOG2 Detection")
cv2.createTrackbar("Min Area", "MOG2 Detection", MIN_AREA, 5000, lambda v: None)
cv2.createTrackbar("Min Ratio", "MOG2 Detection", MIN_RATIO, 300, lambda v: None)

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        field = frame[y:y+h, x:x+w]

        # Detect moving objects
        mask = mog2.apply(field, learningRate=LEARNING_RATE)

        # Find motion contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        display = frame.copy()
        filtered_mask = np.zeros_like(mask)

        # Read slider values
        min_area = cv2.getTrackbarPos("Min Area", "MOG2 Detection")
        min_ratio = cv2.getTrackbarPos("Min Ratio", "MOG2 Detection") / 100

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            # Aspect ratio filter
            bx, by, bw, bh = cv2.boundingRect(contour)
            ratio = bw / bh

            if ratio < min_ratio:
                continue

            # Draw accepted contour on filtered mask
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

            # Draw detected object
            cv2.rectangle(
                display,
                (x+bx, y+by),
                (x+bx+bw, y+by+bh),
                (0, 255, 0), 2
            )

            cv2.putText(
                display,
                f"{int(area)} | R:{ratio:.2f}",
                (x+bx, y+by-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1
            )

    cv2.imshow("MOG2 Detection", display)
    cv2.imshow("Motion Mask", mask)
    cv2.imshow("Filtered Motion Mask", filtered_mask)

    key = cv2.waitKey(0 if paused else 30) & 0xFF

    if key == ord("q"):
        break
    elif key == 32:
        paused = not paused

cap.release()
cv2.destroyAllWindows()
