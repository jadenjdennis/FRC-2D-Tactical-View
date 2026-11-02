
import cv2
import numpy as np

VIDEO_PATH = "assets/match.mov"
START_TIME = 10

# Tracking settings
TRAVEL_SECONDS = 1.5
STUCK_FRAMES = 8
STUCK_DISTANCE = 3
ZONE_MARGIN = 60

# Recovery settings
MIN_HIDDEN_FRAMES = 8
CONFIRM_FRAMES = 4
MIN_AREA_RATIO = 0.20
MAX_AREA_RATIO = 3.0

# Background subtraction
BG_HISTORY = 300
BG_THRESHOLD = 35

# Yellow ball filtering (HSV)
YELLOW_LOW = np.array([15, 65, 65])
YELLOW_HIGH = np.array([40, 255, 255])


def select_box(title, image):
    box = cv2.selectROI(title, image, False, False)
    cv2.destroyWindow(title)
    return tuple(int(v) for v in box)


def center(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2


def near_zone(point, zone, margin):
    x, y = point
    zx, zy, zw, zh = zone

    return (
        zx - margin <= x <= zx + zw + margin
        and zy - margin <= y <= zy + zh + margin
    )


def make_tracker(image, box):
    if hasattr(cv2, "TrackerCSRT_Params"):
        params = cv2.TrackerCSRT_Params()
        params.number_of_scales = 1
        tracker = cv2.TrackerCSRT_create(params)
    else:
        tracker = cv2.legacy.TrackerCSRT_create()

    tracker.init(image, box)
    return tracker


def fixed_box(box, width, height):
    x, y, w, h = [int(v) for v in box]
    cx = x + w // 2
    cy = y + h // 2

    return (
        cx - width // 2,
        cy - height // 2,
        width,
        height
    )


def clamp_box(box, image):
    x, y, w, h = box
    ih, iw = image.shape[:2]

    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))

    return int(x), int(y), int(w), int(h)


def remove_yellow_motion(frame, motion_mask):
    """Remove yellow pixels from the motion mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        YELLOW_LOW,
        YELLOW_HIGH
    )

    cleaned = motion_mask.copy()
    cleaned[yellow_mask > 0] = 0

    return cleaned


def restrict_to_zones(mask, zones):
    """Keep motion only inside the selected recovery zones."""
    zone_mask = np.zeros_like(mask)

    for side, zone in zones:
        x, y, w, h = zone

        cv2.rectangle(
            zone_mask,
            (x, y),
            (x + w, y + h),
            255,
            -1
        )

    return cv2.bitwise_and(mask, zone_mask)


def find_moving_robot(mask, frame, zones, robot_size):
    robot_w, robot_h = robot_size
    robot_area = robot_w * robot_h

    best_box = None
    best_score = -1
    best_side = None

    for side, zone in zones:
        zx, zy, zw, zh = zone

        search_mask = mask[
            zy:zy + zh,
            zx:zx + zw
        ]

        contours, _ = cv2.findContours(
            search_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h

            if area < robot_area * MIN_AREA_RATIO:
                continue

            if area > robot_area * MAX_AREA_RATIO:
                continue

            cx = zx + x + w // 2
            cy = zy + y + h // 2

            width_error = abs(w - robot_w) / robot_w
            height_error = abs(h - robot_h) / robot_h

            score = 1.0 / (
                1.0 + width_error + height_error
            )

            if score > best_score:
                best_score = score
                best_box = clamp_box(
                    (
                        cx - robot_w // 2,
                        cy - robot_h // 2,
                        robot_w,
                        robot_h
                    ),
                    frame
                )
                best_side = side

    return best_box, best_score, best_side


# Open video
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit("Could not open video")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Could not determine FPS")

video.set(
    cv2.CAP_PROP_POS_FRAMES,
    int(START_TIME * fps)
)

success, frame = video.read()

if not success:
    raise SystemExit("Could not read starting frame")


# Initial selections
print("Draw LEFT recovery zone")
left_zone = select_box("Left Recovery Zone", frame)

print("Draw RIGHT recovery zone")
right_zone = select_box("Right Recovery Zone", frame)

print("Select robot")
robot_box = select_box("Select Robot", frame)

if any(
    box[2] <= 0 or box[3] <= 0
    for box in [left_zone, right_zone, robot_box]
):
    raise SystemExit("Selection cancelled")


fixed_width = robot_box[2]
fixed_height = robot_box[3]

tracker = make_tracker(frame, robot_box)

zones = [
    ("LEFT", left_zone),
    ("RIGHT", right_zone)
]

# Background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=BG_HISTORY,
    varThreshold=BG_THRESHOLD,
    detectShadows=False
)

background.apply(frame)

mode = "TRACKING"
positions = []
stuck_count = 0
hidden_frames = 0
travel_frame = 0

start_point = None
exit_zone = None

last_candidate = None
confirm_count = 0
candidate_score = 0

show_motion = False

travel_frames = max(1, int(TRAVEL_SECONDS * fps))

print("Tracking started.")
print("Q = Quit")
print("M = Toggle filtered motion mask")


while True:
    success, frame = video.read()

    if not success:
        break

    display = frame.copy()

    # Background subtraction
    raw_mask = background.apply(
        frame,
        learningRate=0.005
    )

    # Remove yellow ball motion
    motion_mask = remove_yellow_motion(
        frame,
        raw_mask
    )

    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # STRICTLY keep motion inside zones
    motion_mask = restrict_to_zones(
        motion_mask,
        zones
    )

    # Draw recovery zones
    for side, zone in zones:
        zx, zy, zw, zh = zone

        cv2.rectangle(
            display,
            (zx, zy),
            (zx + zw, zy + zh),
            (255, 180, 0),
            2
        )

        cv2.putText(
            display,
            side,
            (zx, max(20, zy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2
        )

    if mode == "TRACKING":
        found, tracked_box = tracker.update(frame)

        if found:
            robot_box = fixed_box(
                tracked_box,
                fixed_width,
                fixed_height
            )

            x, y, w, h = robot_box
            point = (x + w // 2, y + h)

            positions.append(point)
            positions = positions[-10:]

            if len(positions) >= 8:
                old_x, old_y = positions[0]

                movement = (
                    (point[0] - old_x) ** 2
                    + (point[1] - old_y) ** 2
                ) ** 0.5

                if movement < STUCK_DISTANCE:
                    stuck_count += 1
                else:
                    stuck_count = 0

            near_left = near_zone(
                point, left_zone, ZONE_MARGIN
            )

            near_right = near_zone(
                point, right_zone, ZONE_MARGIN
            )

            if (
                stuck_count >= STUCK_FRAMES
                and (near_left or near_right)
            ):
                mode = "PREDICTING"
                hidden_frames = 0
                travel_frame = 0
                start_point = point

                exit_zone = (
                    right_zone if near_left else left_zone
                )

                confirm_count = 0
                last_candidate = None

                print(
                    "Robot hidden. Watching ONLY "
                    "the two recovery zones."
                )

            else:
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        else:
            mode = "SEARCHING"
            hidden_frames = 0
            confirm_count = 0
            last_candidate = None

            print("Tracker lost. Searching recovery zones.")

    elif mode in ("PREDICTING", "SEARCHING"):
        hidden_frames += 1

        if mode == "PREDICTING":
            travel_frame += 1

            progress = min(
                travel_frame / travel_frames,
                1.0
            )

            sx, sy = start_point
            ex, ey = center(exit_zone)

            px = int(sx + (ex - sx) * progress)
            py = int(sy + (ey - sy) * progress)

            cv2.circle(
                display,
                (px, py),
                10,
                (0, 255, 255),
                2
            )

            if progress >= 1.0:
                mode = "SEARCHING"
                print("Searching both recovery zones.")

        if hidden_frames >= MIN_HIDDEN_FRAMES:
            candidate, score, side = find_moving_robot(
                motion_mask,
                frame,
                zones,
                (fixed_width, fixed_height)
            )

            candidate_score = score

            if candidate is not None:
                x, y, w, h = candidate

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 165, 255),
                    2
                )

                if last_candidate is not None:
                    old_x, old_y = center(last_candidate)
                    new_x, new_y = center(candidate)

                    distance = (
                        (new_x - old_x) ** 2
                        + (new_y - old_y) ** 2
                    ) ** 0.5

                    if distance < max(
                        fixed_width, fixed_height
                    ):
                        confirm_count += 1
                    else:
                        confirm_count = 1
                else:
                    confirm_count = 1

                last_candidate = candidate

                if confirm_count >= CONFIRM_FRAMES:
                    tracker = make_tracker(frame, candidate)
                    robot_box = candidate

                    positions = []
                    stuck_count = 0
                    hidden_frames = 0
                    confirm_count = 0
                    last_candidate = None

                    mode = "TRACKING"

                    print(
                        f"Robot reacquired at {side}! "
                        "CSRT resumed."
                    )

            else:
                confirm_count = 0
                last_candidate = None

    # Status overlay
    colors = {
        "TRACKING": (0, 255, 0),
        "PREDICTING": (0, 255, 255),
        "SEARCHING": (255, 200, 0)
    }

    cv2.rectangle(
        display,
        (10, 10),
        (470, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        mode,
        (25, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        colors[mode],
        3
    )

    cv2.putText(
        display,
        f"Motion score: {candidate_score:.2f}",
        (25, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if show_motion:
        cv2.imshow("Filtered Motion", motion_mask)
    else:
        try:
            cv2.destroyWindow("Filtered Motion")
        except cv2.error:
            pass

    cv2.imshow("FRC Robot Tracking", display)

    delay = max(1, int(1000 / fps))
    key = cv2.waitKey(delay) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("m"):
        show_motion = not show_motion


video.release()
cv2.destroyAllWindows()
