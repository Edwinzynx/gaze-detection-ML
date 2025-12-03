import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import os

# Optional: Disable GPU delegate issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

pyautogui.FAILSAFE = False

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# smoothing variables
smooth_x, smooth_y = 0, 0
alpha = 0.25     # smoothing factor

# ESC handling
last_esc_time = 0

# Blink detection
blink_times = []
freeze_mode = False   # stop cursor & reset head position

# Sensitivity (gain multiplier)
GAIN = 4.5     # Increase if needed (3–7 ideal)

# Baseline (eye center when not frozen)
baseline_x = None
baseline_y = None

# Baseline auto-reset timing
baseline_reset_interval = 60          # reset every 1 minute
last_baseline_time = time.time()      # track when last baseline was set


def EAR(eye):
    vertical = abs(eye[0].y - eye[1].y)
    horizontal = abs(eye[2].x - eye[3].x)
    return vertical / horizontal


while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # ----- IRIS CENTER -----
        iris_ids = [468, 469, 470, 471]
        iris_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in iris_ids])
        cx, cy = np.mean(iris_pts, axis=0)

        # Initialize baseline once
        if baseline_x is None:
            baseline_x = cx
        if baseline_y is None:
            baseline_y = cy

        # ---------------------------------------
        # AUTO RESET BASELINE EVERY 60 SECONDS
        # ---------------------------------------
        elapsed = time.time() - last_baseline_time

        if elapsed > baseline_reset_interval:
            baseline_x = cx
            baseline_y = cy
            last_baseline_time = time.time()
            print("Baseline auto-reset!")

        # ---------------------------------------
        # RELATIVE POSITION WITH SENSITIVITY
        # ---------------------------------------
        rel_x = (cx - baseline_x) * GAIN
        rel_y = (cy - baseline_y) * GAIN

        # Map movement to entire screen
        target_x = np.interp(rel_x, [-150, 150], [0, screen_w])
        target_y = np.interp(rel_y, [-100, 100], [0, screen_h])

        # smoothing
        smooth_x = smooth_x + alpha * (target_x - smooth_x)
        smooth_y = smooth_y + alpha * (target_y - smooth_y)

        # move cursor only if NOT frozen
        if not freeze_mode:
            pyautogui.moveTo(smooth_x, smooth_y)

        # ----- BLINK DETECTION VIA EAR -----
        left_eye_ids = [159, 145, 33, 133]   # upper, lower, left, right
        eye = [lm[i] for i in left_eye_ids]
        ear = EAR(eye)

        if ear < 0.18:
            blink_times.append(time.time())

            # keep last 1.5 sec blinks
            blink_times = [t for t in blink_times if time.time() - t < 1.5]

            # ----- TRIPLE BLINK = TOGGLE FREEZE MODE -----
            if len(blink_times) == 3:
                freeze_mode = not freeze_mode

                if freeze_mode:
                    print("FREEZE ON (pause cursor & allow head reset)")
                else:
                    print("FREEZE OFF (recenter)")
                    time.sleep(0.15)  # allow eyes to open
                    baseline_x = cx
                    baseline_y = cy
                    last_baseline_time = time.time()

                blink_times = []
                time.sleep(0.25)
                continue

            # ----- SINGLE BLINK CLICK (only when not frozen) -----
            if not freeze_mode:
                pyautogui.click()
                print("Click")
                time.sleep(0.25)

        # draw iris points (debug)
        for p in iris_pts:
            cv2.circle(frame, tuple(p.astype(int)), 2, (0,255,0), -1)

    # ---- countdown timer display ----
    remaining = int(baseline_reset_interval - (time.time() - last_baseline_time))
    if remaining < 0:
        remaining = 0

    cv2.putText(frame, f"Auto-reset in: {remaining}s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    # ----- DOUBLE ESC EXIT -----
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        now = time.time()
        if now - last_esc_time < 1.2:
            break
        last_esc_time = now

    cv2.imshow("Eye Control", frame)

cam.release()
cv2.destroyAllWindows()
