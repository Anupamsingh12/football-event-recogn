'''
Tracks players in a football match using YOLOv8 and DeepSORT.
Assigns teams based on jersey color clustering.
'''

from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from deep_sort_realtime.deepsort_tracker import DeepSort
import yt_dlp
# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Load video
# cap = cv2.VideoCapture("video.mp4")  # or 0 for webcam
width, height = 960, 540

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50)

# Variables
jersey_colors = {}        # player_id -> jersey dominant color
player_to_team = {}       # player_id -> team label
team_centers = None       # cluster centers for teams
clustering_done = False

def assign_team_with_hard_threshold(new_color, team_centers, max_distance=10):
    distances = np.linalg.norm(team_centers - new_color, axis=1)
    min_dist = np.min(distances)
    if min_dist > max_distance:
        return None  # Too far from both teams, reject
    return np.argmin(distances)  # Return team 0 or 1

def is_trusted_team_color(dominant_color_rgb):
    # Convert RGB to HSV
    dominant_color = np.uint8([[dominant_color_rgb]])
    hsv_color = cv2.cvtColor(dominant_color, cv2.COLOR_RGB2HSV)[0][0]

    h, s, v = hsv_color  # Hue, Saturation, Value

    # Example trusted zones
    if (0 <= h <= 20 or 160 <= h <= 180) and s > 100:  # Red shades
        return True
    if (90 <= h <= 140) and s > 50:  # Blue shades
        return True
    if s < 40 and v > 200:  # 🤍 White jerseys: low saturation, high brightness
        return True

    return False  # Otherwise not trusted (yellow, black kits)

    # Convert RGB to HSV
    dominant_color = np.uint8([[dominant_color_rgb]])
    hsv_color = cv2.cvtColor(dominant_color, cv2.COLOR_RGB2HSV)[0][0]

    h, s, v = hsv_color  # Hue, Saturation, Value

    # Example trusted zones (adjust based on your kits)
    if (0 <= h <= 20 or 160 <= h <= 180) and s > 100:  # Red shades
        return True
    if (90 <= h <= 140) and s > 50:  # Blue/white shades
        return True

    return False  # Otherwise not trusted (referee, yellow/black kits)

youtube_url = "https://www.youtube.com/watch?v=uc19uwliL3k&t=126s"

def get_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4][height<=480]',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

stream_url = get_stream_url(youtube_url)
cap = cv2.VideoCapture(stream_url)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    result = model(frame)[0]
    annotated = frame.copy()

    ball_center = None
    detections_for_tracker = []

    # Prepare detections
    for box in result.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # Person (player)
            confidence = float(box.conf[0])
            width_box = x2 - x1
            height_box = y2 - y1
            aspect_ratio = height_box / (width_box + 1e-6)
            box_area = width_box * height_box

            # ✅ Strong Filtering
            if confidence > 0.7 and width_box > 15 and height_box > 30 and 1.2 < aspect_ratio < 3.5 and box_area < 20000:
                detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'player'))

        elif cls == 32:  # Football
            width_ball = x2 - x1
            height_ball = y2 - y1
            ball_area = width_ball * height_box

            if 300 < ball_area < 5000:  # Acceptable ball size
                ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(annotated, ball_center, 5, (0, 0, 255), -1)
                cv2.putText(annotated, "Ball", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Update tracks
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    player_centers = {}
    player_boxes = {}
    jersey_features = []
    jersey_player_ids = []

    if tracks:
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            player_boxes[track_id] = (x1, y1, x2, y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            player_centers[track_id] = center

            # Crop jersey (upper half of body)
            jersey_crop = frame[y1:int((y1 + y2) / 2), x1:x2]
            if jersey_crop.size > 0:
                crop = jersey_crop.reshape((-1, 3))
                kmeans = KMeans(n_clusters=1, n_init=5)
                kmeans.fit(crop)
                dominant_color = kmeans.cluster_centers_[0]
                jersey_colors[track_id] = dominant_color
                jersey_features.append(dominant_color)
                jersey_player_ids.append(track_id)

    # Cluster into two teams ONLY ONCE
    if not clustering_done and len(jersey_features) >= 10:
        kmeans_team = KMeans(n_clusters=2, n_init=10)
        labels = kmeans_team.fit_predict(jersey_features)

        team_centers = kmeans_team.cluster_centers_  # Save cluster centers!

        for idx, label in enumerate(labels):
            player_to_team[jersey_player_ids[idx]] = label

        clustering_done = True

    # Draw players
    for player_id, center in player_centers.items():
        x1, y1, x2, y2 = player_boxes[player_id]

        # Predict team if player not assigned yet
        if player_id not in player_to_team and clustering_done:
            if player_id in jersey_colors:
                dominant_color = jersey_colors[player_id]

                # 💥 Check if jersey color is trusted first
                # if not is_trusted_team_color(dominant_color):
                #     continue  # Skip referee/fan

                predicted_team = assign_team_with_hard_threshold(dominant_color, team_centers, max_distance=20)
                if predicted_team is not None:
                    player_to_team[player_id] = predicted_team

        # Only draw players if they have a valid team
        if player_id not in player_to_team:
            continue

        team_label = player_to_team[player_id]

        if team_label == 0:
            color = (255, 0, 0)  # Blue for Team A
        else:
            color = (0, 255, 0)  # Green for Team B

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"Player {player_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ After drawing all players, show the frame
    cv2.imshow("Football Tracker - Teams Locked", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()