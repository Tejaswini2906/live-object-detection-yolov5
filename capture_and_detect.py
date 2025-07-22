import cv2
import os
import time
from glob import glob

# Step 1: Capture image
cap = cv2.VideoCapture(0)
print("📸 Opening webcam...")
time.sleep(2)
ret, frame = cap.read()

if ret:
    img_path = "live_capture.jpg"
    cv2.imwrite(img_path, frame)
    print("✅ Image saved as", img_path)
else:
    print("❌ Failed to capture image")

cap.release()
cv2.destroyAllWindows()

# Step 2: Run YOLOv5 detection
print("🔍 Running YOLOv5 detection...")
os.system(f"python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source {img_path}")

# Step 3: Locate the latest 'exp*' folder inside runs/detect
exp_folders = sorted(glob("runs/detect/exp*"), key=os.path.getmtime, reverse=True)
if exp_folders:
    latest_folder = exp_folders[0]
    result_path = os.path.join(latest_folder, "live_capture.jpg")

    if os.path.exists(result_path):
        img = cv2.imread(result_path)
        cv2.imshow("🧠 Detected Image", img)
        print(f"🖼️ Showing result from: {result_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Image not found in {latest_folder}")
else:
    print("❌ No detection folders found.")
