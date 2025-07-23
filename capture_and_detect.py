import cv2
import os
import time
from glob import glob

# Step 1: Open webcam and capture live video frames, run detection on each frame
cap = cv2.VideoCapture(0)
print("📸 Opening webcam...")

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🔍 Running YOLOv5 detection on live video...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Save frame temporarily for detection input
        tmp_img_path = "tmp_live.jpg"
        cv2.imwrite(tmp_img_path, frame)

        # Run YOLOv5 detect.py on the current frame image
        # This will create/overwrite folders in runs/detect/exp* with results
        os.system(f"python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source {tmp_img_path} --hide-labels --hide-conf --save-txt --save-conf")

        # Find latest exp folder created by YOLOv5 detection
        exp_folders = sorted(glob("runs/detect/exp*"), key=os.path.getmtime, reverse=True)
        if exp_folders:
            latest_folder = exp_folders[0]
            result_img_path = os.path.join(latest_folder, "tmp_live.jpg")  # detected image saved with same name

            if os.path.exists(result_img_path):
                detected_img = cv2.imread(result_img_path)
                cv2.imshow("🧠 YOLOv5 Detection (Press 'q' to quit)", detected_img)
            else:
                print(f"❌ Detected image not found in {latest_folder}")
                cv2.imshow("Webcam", frame)
        else:
            print("❌ No detection folders found.")
            cv2.imshow("Webcam", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Clean up temp image if exists
    if os.path.exists("tmp_live.jpg"):
        os.remove("tmp_live.jpg")
