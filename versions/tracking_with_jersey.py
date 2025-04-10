from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load YOLOv8 model (you can also try "yolov8n.pt", "yolov8s.pt", etc.)
model = YOLO("yolov8n.pt")

# Video capture
cap = cv2.VideoCapture("video.mp4")  # change to 0 for webcam
width, height = 640, 384

# Initialize team clustering
team_colors = {}
last_possession_team = None

def get_dominant_color(crop):
    crop = crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=5)
    kmeans.fit(crop)
    return kmeans.cluster_centers_[0]

def get_team_labels(player_crops):
    if len(player_crops) < 2:
        return [0] * len(player_crops)
    features = [get_dominant_color(crop) for crop in player_crops]
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    result = model(frame)[0]
    annotated = frame.copy()

    ball_center = None
    player_boxes = []
    jersey_crops = []

    for box in result.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if cls == 32:  # ball
            ball_center = center
            cv2.circle(annotated, ball_center, 5, (0, 0, 255), -1)
            cv2.putText(annotated, "Ball", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        elif cls == 0:  # person
            player_boxes.append(((x1, y1, x2, y2), center))
            jersey = frame[y1:int((y1 + y2) / 2), x1:x2]
            if jersey.size > 0:
                jersey_crops.append(jersey)

    team_labels = get_team_labels(jersey_crops)

    # Track ball possession
    possession_team = None
    if ball_center and player_boxes:
        min_dist = float("inf")
        closest_player_idx = -1
        for i, (_, center) in enumerate(player_boxes):
            dist = np.linalg.norm(np.array(ball_center) - np.array(center))
            if dist < min_dist:
                min_dist = dist
                closest_player_idx = i

        if closest_player_idx >= 0:
            (x1, y1, x2, y2), _ = player_boxes[closest_player_idx]
            possession_team = team_labels[closest_player_idx]
            color = (0, 255, 0) if possession_team == 0 else (255, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, "Possession", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Detect possession change
    if possession_team is not None and possession_team != last_possession_team:
        team_name = "Team White" if possession_team == 0 else "Team Maroon"
        cv2.putText(annotated, f"🔁 Possession: {team_name}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        last_possession_team = possession_team

    cv2.imshow("Football Tracker", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
