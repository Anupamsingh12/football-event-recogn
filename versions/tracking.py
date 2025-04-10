from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("video.mp4")  # Change this to your video path

# Frame dimensions
width = 640
height = 384

# Possession tracking settings
POSSESSION_DISTANCE = 40
last_possession_team = None
possession_change_timer = 0

def get_dominant_color(image):
    avg_color = np.mean(image.reshape(-1, 3), axis=0)
    return avg_color

def classify_team(color_bgr):
    b, g, r = color_bgr
    if r > 150 and r > g + 40:
        return "Red Team"
    elif r > 180 and g > 180 and b > 180:
        return "White Team"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    results = model(frame)[0]
    annotated = frame.copy()

    ball_center = None
    player_boxes = []

    for box in results.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if cls == 32:  # Ball
            ball_center = center
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, "Ball", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        elif cls == 0:  # Player
            player_boxes.append(((x1, y1, x2, y2), center))

    possession_team = None

    if ball_center and player_boxes:
        min_dist = float("inf")
        closest_player = None
        closest_crop = None

        for (bbox, center) in player_boxes:
            dist = np.linalg.norm(np.array(ball_center) - np.array(center))
            if dist < min_dist:
                min_dist = dist
                closest_player = bbox
                x1, y1, x2, y2 = bbox
                pad = 5
                crop = frame[max(y1+pad, 0):min(y2-pad, height),
                             max(x1+pad, 0):min(x2-pad, width)]
                closest_crop = crop

        if closest_player and closest_crop is not None:
            avg_color = get_dominant_color(closest_crop)
            possession_team = classify_team(avg_color)

            x1, y1, x2, y2 = closest_player
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"Has Ball - {possession_team}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Update possession text
    if possession_team and possession_team != last_possession_team:
        if last_possession_team is not None:
            possession_change_timer = 30
        last_possession_team = possession_team

    text = f"Possession: {last_possession_team or 'None'}"
    if possession_change_timer > 0:
        text += "  |  Possession Changed!"
        possession_change_timer -= 1

    # Draw info on bottom
    cv2.rectangle(annotated, (0, height - 30), (width, height), (0, 0, 0), -1)
    cv2.putText(annotated, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Show live frame
    cv2.imshow("Football Possession Detection", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
