import os
import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
from collections import deque
# added possession tracker..
# ---------------------- Team Classifier ---------------------- #
class TeamClassifier:
    def __init__(self, calibrator_dir="calibrator"):
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.trained = False
        self._train(calibrator_dir)

    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        h, w = image.shape[:2]
        center_crop = image[int(h * 0.2):int(h * 0.8), int(w * 0.3):int(w * 0.7)]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _train(self, calibrator_dir):
        features, labels = [], []
        for team in ["teamA", "teamB"]:
            label = 0 if team == "teamA" else 1
            folder = os.path.join(calibrator_dir, team)
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    feat = self.extract_color_histogram(img)
                    features.append(feat)
                    labels.append(label)
        if features:
            self.knn.fit(features, labels)
            self.trained = True

    def predict(self, image):
        if not self.trained or image.size == 0:
            return "Team A"
        image = cv2.resize(image, (64, 64))
        feat = self.extract_color_histogram(image)
        label = self.knn.predict([feat])[0]
        return "Team A" if label == 0 else "Team B"

# ---------------------- Setup ---------------------- #
team_classifier = TeamClassifier("calibrator")
model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("video.mp4")  # or YouTube stream

BALL_CLASS_ID = 32
draw_enabled = False
ball_path = deque(maxlen=30)
last_possessor = None
last_possessing_team = None
last_distance = None
draw_path = False

# ---------------------- Loop ---------------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        draw_enabled = True

    if not draw_enabled:
        cv2.imshow("Football Stream", frame)
        continue

    results = model(frame)[0]
    ball_pos = None
    players = []

    # Detect players and ball
    for box in results.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:  # Person
            crop = frame[y1:y2, x1:x2]
            team = team_classifier.predict(crop)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            players.append({'box': (x1, y1, x2, y2), 'team': team, 'center': center})
        elif cls == BALL_CLASS_ID:
            ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Draw players
    for p in players:
        x1, y1, x2, y2 = p['box']
        color = (0, 255, 0) if p['team'] == "Team A" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, p['team'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if ball_pos:
        cv2.circle(frame, ball_pos, 6, (0, 255, 255), -1)
        ball_path.append(ball_pos)

        # Draw ball trail if enabled
        if draw_path:
            for i in range(1, len(ball_path)):
                cv2.line(frame, ball_path[i - 1], ball_path[i], (0, 255, 255), 2)

    # Possession logic
    if ball_pos and players:
        dists = [np.linalg.norm(np.array(ball_pos) - np.array(p['center'])) for p in players]
        min_idx = np.argmin(dists)
        possessor = players[min_idx]
        possessing_team = possessor['team']
        dist = dists[min_idx]

        if last_possessor is None:
            last_possessor = min_idx
            last_possessing_team = possessing_team
            last_distance = dist
        else:
            # Check if current possessor is same but now far from ball
            if min_idx == last_possessor and dist > last_distance + 30:
                draw_path = True
                cv2.putText(frame, f"{possessing_team} passed the ball", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            elif min_idx != last_possessor:
                draw_path = False
                ball_path.clear()
                cv2.putText(frame, f"Possession changed to {possessing_team}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            last_possessor = min_idx
            last_possessing_team = possessing_team
            last_distance = dist

        cv2.putText(frame, f"Possession: {possessing_team}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Football Stream", frame)

cap.release()
cv2.destroyAllWindows()
