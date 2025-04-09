import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

width, height = 640, 384
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 40, 255])
RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])

last_possession = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    results = model(frame)[0]
    annotated = frame.copy()
    ball_center = None
    players = []

    for box in results.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if cls == 32:  # Ball
            ball_center = center
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, "Ball", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        elif cls == 0:  # Player
            # Focus on upper third of body
            upper_y2 = y1 + (y2 - y1) // 3
            upper_body = frame[y1:upper_y2, x1:x2]
            hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

            mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
            mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
            mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            white_score = cv2.countNonZero(mask_white)
            red_score = cv2.countNonZero(mask_red)

            team = "Unknown"
            color = (100, 100, 100)
            if white_score > red_score and white_score > 80:
                team = "Team White"
                color = (255, 255, 255)
            elif red_score > 80:
                team = "Team Red"
                color = (0, 0, 255)

            players.append(((x1, y1, x2, y2), center, team))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, team, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Possession logic
    if ball_center and players:
        min_dist = float("inf")
        possessor_team = None
        for (bbox, center, team) in players:
            dist = np.linalg.norm(np.array(ball_center) - np.array(center))
            if dist < min_dist:
                min_dist = dist
                possessor_team = team

        if possessor_team and possessor_team != "Unknown":
            last_possession = possessor_team

    # Show possession on screen
    cv2.putText(annotated, f"Possession: {last_possession}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Football Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
