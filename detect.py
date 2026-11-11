import cv2
import numpy as np
import csv
from heatmap import generate_heatmap

# These will be the default values at the start of the program
VIDEO = "assets/match6.mp4"
START_TIME = 10

MIN_AREA = 500
MIN_RATIO = 80
LEARNING_RATE = 0.0005

# this should be 2 pixels per inch
# We could lower this if the preview ends up being too big/small
W = 1302
H = 635

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

ret, frame = cap.read()

# this measure will hopefully help it focus on the field
x, y, w, h = cv2.selectROI("Select Field", frame)
cv2.destroyWindow("Select Field")

# Aarmaan: I added this 4 corners selector thing to convert the camera view to top down coords
# Make sure to click in order: top left, top right, bottom right, bottom left <<<<<< 
corners = []

def click(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append([mx - x, my - y])

cv2.namedWindow("Corners")
cv2.setMouseCallback("Corners", click)

while len(corners) < 4:
    preview = frame.copy()

    for px, py in corners:
        cv2.circle(preview, (px + x, py + y), 5, (0, 255, 0), -1)

    cv2.imshow("Corners", preview)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled")

cv2.destroyWindow("Corners")

# This should map the corners to a rectangle to make jadens life a little easier
src = np.float32(corners)

dst = np.float32([
    [0, 0],
    [W - 1, 0],
    [W - 1, H - 1],
    [0, H - 1]
])

matrix = cv2.getPerspectiveTransform(src, dst)

# FOR JADEN: This is not the modle learning rate, for MOG2 the low learning rate is so robots don't disappear when they momentarily pause
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=35,
    detectShadows=False
)

cv2.namedWindow("Detection")

# These are for sliders, they should be easier to adjust than our typing box thing earlier, & we can play around with them live
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

            # This should make it focus on the field crop
            field = frame[y:y+h, x:x+w]
            mask = mog2.apply(field, learningRate=LEARNING_RATE)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            display = frame.copy()
            filtered = np.zeros_like(mask)

            # I'm also adding making a blank image where the mapped robot dots will show up view, this is just for debugging to see the if the tracker is working
            top_down = np.zeros((H, W, 3), dtype=np.uint8)

            area_min = cv2.getTrackbarPos("Min Area", "Detection")
            ratio_slider = cv2.getTrackbarPos("Min Ratio", "Detection")
            ratio_min = ratio_slider / 100

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            for contour in contours:

                area = cv2.contourArea(contour)

                # ignoring smaller moving objects should help us ignore all the little moving balls
                if area < area_min:
                    continue

                bx, by, bw, bh = cv2.boundingRect(contour)
                ratio = bw / bh

                # humans are taller than they are wide, which is the oposite for FRC robots, sooo hopefully this stops following the humans
                if ratio < ratio_min:
                    continue

                # idk who decided to call it countours instead of somethign simple, but this is an example from the docs & it works so ig
                cv2.drawContours(filtered, [contour], -1, 255, -1)

                cv2.rectangle(
                    display,
                    (x + bx, y + by),
                    (x + bx + bw, y + by + bh),
                    (0, 255, 0), 2
                )

                
                robot_x = bx + bw / 2
                robot_y = by + bh

                point = np.float32([[[robot_x, robot_y]]])

                # !!!!!!JADEN!!! everything below this is the field coord mapping and csv stuff so change anything u need to get heatmap working
                mapped = cv2.perspectiveTransform(point, matrix)
                fx = mapped[0][0][0]
                fy = mapped[0][0][1]

                # Jaden: Yo Aarmaan I added this to try and skip the humans that might still get picked up, it should ignore the boxes that bassically touch the bottom
                if fx >= 0 and fx < W and fy >= 0 and fy < H:

                    dot_x = int(fx)
                    dot_y = int(fy)

                    cv2.circle(
                        top_down,
                        (dot_x, dot_y),
                        5, (0, 0, 255), -1
                    )

                    if recording:
                        writer.writerow([
                            round(time_sec, 3),
                            frame_num,
                            round(float(fx)),
                            round(float(fy))
                        ])

        if recording:
            status = "RECORDING ON"
            color = (0, 0, 255) # Thank you for making this red its so much easier to read now 
        else:
            status = "RECORDING OFF"
            color = (150, 150, 150)

        cv2.putText(
            display, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )


        # I'm showing all the masks for the vid
        cv2.imshow("Detection", display)
        cv2.imshow("Motion Mask", mask)
        cv2.imshow("Filtered Motion Mask", filtered)
        cv2.imshow("Top-Down Preview", top_down)

        
        if paused:
            delay = 0
        else:
            #JADEN: CHANGE THIS VAL IF YOU WANNA MAKE IT GO FASTER <---
            delay = 1

        key = cv2.waitKey(delay) & 0xFF

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

# Jadens code takes the CSV and makes the actual heatmap
generate_heatmap()
