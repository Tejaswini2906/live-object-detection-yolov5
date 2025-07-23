import PySimpleGUI as sg
import cv2
import torch
import os
from pathlib import Path
from datetime import datetime
from threading import Thread
import subprocess
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (non_max_suppression, check_img_size, LOGGER, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# Folders
os.makedirs("runs/detect_log/videos", exist_ok=True)
os.makedirs("runs/detect_log/logs", exist_ok=True)

# Load YOLOv5 model
weights = 'yolov5s.pt'
data = 'data/coco128.yaml'
device = select_device('')
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz_checked = check_img_size([640, 640], s=stride)
model.warmup(imgsz=(1, 3, *imgsz_checked))

# GUI Layout
layout = [
    [sg.Text("YOLOv5 Real-Time Video Detection", font=("Helvetica", 16))],
    [sg.Button("Start Live Detection", size=(20, 1), button_color=('white', 'green'))],
    [sg.Text("Previous Detection Videos:", font=("Helvetica", 12))],
    [sg.Listbox(values=[], size=(60, 10), key="-VIDEO_LIST-", enable_events=True)],
    [sg.Button("Play Video"), sg.Button("View Log"), sg.Button("Delete Selected Video", button_color=('white', 'firebrick'))],
    [sg.Button("Refresh List"), sg.Exit()]
]

window = sg.Window("YOLOv5 Detection GUI", layout, finalize=True)

# Update video list
def update_video_list():
    video_dir = Path("runs/detect_log/videos")
    videos = sorted([f.name for f in video_dir.glob("*.mp4")], reverse=True)
    window["-VIDEO_LIST-"].update(videos)

# Live detection with recording and logging
def detect_and_record_video():
    dataset = LoadStreams("0", img_size=imgsz_checked, stride=stride, auto=pt)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"detect_{now}.mp4"
    video_path = f"runs/detect_log/videos/{video_filename}"
    log_path = f"runs/detect_log/logs/{video_filename.replace('.mp4', '.txt')}"

    log_file = open(log_path, "a")
    writer = None

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im[None]
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_file.write(f"[{timestamp}] {label}\n")
                    print(f"[{timestamp}] {label}")
            im0 = annotator.result()

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = im0.shape[:2]
                writer = cv2.VideoWriter(video_path, fourcc, 20, (w, h))

            writer.write(im0)
            cv2.imshow("Live Detection", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                log_file.close()
                writer.release()
                cv2.destroyAllWindows()
                update_video_list()
                return

# Start
update_video_list()

# GUI Event Loop
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    elif event == "Start Live Detection":
        Thread(target=detect_and_record_video, daemon=True).start()

    elif event == "Refresh List":
        update_video_list()

    elif event == "Delete Selected Video":
        selected = values["-VIDEO_LIST-"]
        if selected:
            video_file = selected[0]
            video_path = os.path.join("runs/detect_log/videos", video_file)
            log_file = os.path.join("runs/detect_log/logs", video_file.replace(".mp4", ".txt"))
            try:
                os.remove(video_path)
                if os.path.exists(log_file):
                    os.remove(log_file)
                sg.popup("Deleted successfully!")
                update_video_list()
            except Exception as e:
                sg.popup("Error deleting:", str(e))
        else:
            sg.popup("Select a video to delete.")

    elif event == "Play Video":
        selected = values["-VIDEO_LIST-"]
        if selected:
            video_file = os.path.join("runs/detect_log/videos", selected[0])
            os.startfile(video_file)
        else:
            sg.popup("Select a video to play.")

    elif event == "View Log":
        selected = values["-VIDEO_LIST-"]
        if selected:
            log_path = os.path.join("runs/detect_log/logs", selected[0].replace(".mp4", ".txt"))
            if os.path.exists(log_path):
                subprocess.Popen(['notepad.exe', log_path])
            else:
                sg.popup("Log not found for selected video.")
        else:
            sg.popup("Select a video to view log.")

window.close()
