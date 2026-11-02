
import cv2

VIDEO_PATH = "assets/match.mov"

# Open the match video
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open video!")
    exit()

START_TIME = 10

fps = video.get(cv2.CAP_PROP_FPS)
start_frame = int(START_TIME * fps)

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

success, frame = video.read()

if not success:
    print("Could not read the first frame!")
    video.release()
    exit()

# Select the robot to track
print("Draw a box around one robot, then press ENTER.")
box = cv2.selectROI("Select Robot", frame, False, False)
cv2.destroyWindow("Select Robot")

if box[2] == 0 or box[3] == 0:
    print("No robot selected.")
    video.release()
    exit()

# Create and initialize the tracker
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, box)

# Track the robot through the video
while True:
    success, frame = video.read()

    if not success:
        break

    found, box = tracker.update(frame)

    if found:
        x, y, w, h = [int(value) for value in box]

        # Draw a box around the robot
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the bottom-center of the robot
        center_x = x + w // 2
        center_y = y + h

        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    else:
        cv2.putText(frame, "TRACKING LOST", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("FRC Robot Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()