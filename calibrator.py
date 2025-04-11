
import cv2
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Setup calibration directories
os.makedirs("calibrator/teamA", exist_ok=True)
os.makedirs("calibrator/teamB", exist_ok=True)

# YouTube stream
import yt_dlp

youtube_url = "https://www.youtube.com/watch?v=uc19uwliL3k&t=126s"

def get_stream_url(url):
    ydl_opts = {'quiet': True, 'format': 'best[ext=mp4][height<=480]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

stream_url = get_stream_url(youtube_url)
# cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture("video.mp4")  # Use local video for testing

counter_a, counter_b = 0, 0
current_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    players = [box for box in results.boxes if int(box.cls) == 0]

    if current_box is None and players:
        current_box = players[0]

    annotated = frame.copy()

    if current_box is not None:
        x1, y1, x2, y2 = map(int, current_box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annotated, "LOCKED - Press A or B", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Calibration Lock Mode", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and current_box is not None:
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(f"calibrator/teamA/player_{counter_a}.jpg", crop)
            print(f"Saved Team A player {counter_a}")
            counter_a += 1
        current_box = None

    elif key == ord('b') and current_box is not None:
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(f"calibrator/teamB/player_{counter_b}.jpg", crop)
            print(f"Saved Team B player {counter_b}")
            counter_b += 1
        current_box = None
    elif key == ord('s'):
        current_box = None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

