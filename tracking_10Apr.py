import os
import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

# ---------------------- Calibration Function ---------------------- #
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    h, w = image.shape[:2]
    center_crop = image[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]  # torso area
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def predict_team_knn(player_crop, calibrator_dir="calibrator"):
    features = []
    labels = []

    for team in ["teamA", "teamB"]:
        label = 0 if team == "teamA" else 1
        folder = os.path.join(calibrator_dir, team)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                feat = extract_color_histogram(img)
                features.append(feat)
                labels.append(label)

    if not features:
        return "Unknown"

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)

    resized_crop = cv2.resize(player_crop, (64, 64))
    feat = extract_color_histogram(resized_crop)
    label = knn.predict([feat])[0]

    return "Team A" if label == 0 else "Team B"

# ---------------------- YouTube Stream Setup ---------------------- #
youtube_url = "https://www.youtube.com/watch?v=uc19uwliL3k&t=126s"

def get_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4][height<=720]',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

stream_url = get_stream_url(youtube_url)
cap = cv2.VideoCapture(stream_url)

# ---------------------- Load YOLOv8 Model ---------------------- #
model = YOLO("yolov8s.pt")  # change to "yolov8s.pt" if needed

# ---------------------- Processing Loop ---------------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls)
        if cls != 0:  # class 0 = person
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            continue

        team = predict_team_knn(player_crop)
        color = (0, 255, 0) if team == "Team A" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imshow("YouTube Football Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
