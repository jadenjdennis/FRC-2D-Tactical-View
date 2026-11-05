
import cv2
import numpy as np
import csv

VIDEO = "assets/match6.mp4"
START_TIME = 10
MIN_AREA = 500
MIN_RATIO = 80
LEARNING_RATE = 0.0005

# 2026 REBUILT field proportions (2 pixels per inch)
W, H = 1302, 635

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Could not read video")

x, y, w, h = cv2.selectROI("Select Field", frame)
cv2.destroyWindow("Select Field")
if not w or not h:
    raise SystemExit("No field selected")

# Click four field corners: TL, TR, BR, BL
corners = []

def click(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append([mx - x, my - y])

cv2.namedWindow("Corners")
cv2.setMouseCallback("Corners", click)

while len(corners) < 4:
    preview = frame.copy()
    for px, py in corners:
        cv2.circle(preview, (px + x, py + y), 5, (0, 255, 0), -1)
    cv2.imshow("Corners", preview)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        raise SystemExit("Cancelled")

cv2.destroyWindow("Corners")

src = np.float32(corners)
dst = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])
matrix = cv2.getPerspectiveTransform(src, dst)

mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=35, detectShadows=False
)

cv2.namedWindow("Detection")
cv2.createTrackbar("Min Area", "Detection", MIN_AREA, 5000, lambda v: None)
cv2.createTrackbar("Min Ratio", "Detection", MIN_RATIO, 300, lambda v: None)

paused = False
recording = False

with open("robot_positions.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time_sec", "frame", "field_x", "field_y"])

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            mask = mog2.apply(
                frame[y:y+h, x:x+w], learningRate=LEARNING_RATE
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            display = frame.copy()
            filtered = np.zeros_like(mask)
            top_down = np.zeros((H, W, 3), dtype=np.uint8)

            area_min = cv2.getTrackbarPos("Min Area", "Detection")
            ratio_min = cv2.getTrackbarPos("Min Ratio", "Detection") / 100

            for contour in contours:
                if cv2.contourArea(contour) < area_min:
                    continue

                bx, by, bw, bh = cv2.boundingRect(contour)
                if bw / bh < ratio_min:
                    continue

                cv2.drawContours(filtered, [contour], -1, 255, -1)
                cv2.rectangle(
                    display, (x+bx, y+by),
                    (x+bx+bw, y+by+bh), (0, 255, 0), 2
                )

                point = np.float32([[[bx + bw/2, by + bh]]])
                fx, fy = cv2.perspectiveTransform(point, matrix)[0][0]

                if 0 <= fx < W and 0 <= fy < H:
                    cv2.circle(top_down, (int(fx), int(fy)), 5, (0, 0, 255), -1)

                    if recording:
                        writer.writerow([
                            round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 3),
                            int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            round(float(fx)), round(float(fy))
                        ])

        status = "RECORDING ON" if recording else "RECORDING OFF"
        cv2.putText(display, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (150, 150, 150), 2)

        cv2.imshow("Detection", display)
        cv2.imshow("Motion Mask", mask)
        cv2.imshow("Filtered Motion Mask", filtered)
        cv2.imshow("Top-Down Preview", top_down)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:
            paused = not paused
        elif key == ord("s"):
            recording = not recording
            file.flush()
            print("Recording:", "ON" if recording else "OFF")

cap.release()
cv2.destroyAllWindows()
print("Saved robot_positions.csv")
