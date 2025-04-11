import os
import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

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

# ---------------------- YouTube Stream ---------------------- #
def get_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4][height<=720]',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# ---------------------- Setup ---------------------- #
team_classifier = TeamClassifier("calibrator")
model = YOLO("yolov8s.pt")
# stream_url = get_stream_url("https://www.youtube.com/watch?v=...")
# cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture("video.mp4")  # for testing

last_possessing_team = None
last_possessor_idx = None
BALL_CLASS_ID = 32

# ---------------------- Main Loop ---------------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    ball_pos = None
    players = []

    # Detect ball & collect player info
    for box in results.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:  # Person
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            crop = frame[y1:y2, x1:x2]
            players.append({'box': (x1, y1, x2, y2), 'center': center, 'crop': crop})
        elif cls == BALL_CLASS_ID:
            ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, ball_pos, 6, (0, 255, 255), -1)

    # Only classify closest player
    if ball_pos and players:
        dists = [np.linalg.norm(np.array(ball_pos) - np.array(p['center'])) for p in players]
        min_idx = int(np.argmin(dists))
        possessor = players[min_idx]
        team = team_classifier.predict(possessor['crop'])

        # Draw only closest player's box
        x1, y1, x2, y2 = possessor['box']
        color = (0, 255, 0) if team == "Team A" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display possession
        cv2.putText(frame, f"Possession: {team}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Possession change logic
        if last_possessing_team != team:
            cv2.putText(frame, f"Possession changed to {team}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            last_possessing_team = team
            last_possessor_idx = min_idx

        # Pass logic (within same team, different player)
        elif last_possessor_idx != min_idx:
            cv2.putText(frame, f"{team} passed the ball", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            last_possessor_idx = min_idx

    cv2.imshow("Football Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
