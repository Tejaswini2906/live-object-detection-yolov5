# YOLOv5 Real-Time Object Detection with GUI

## Description
A complete real-time object detection system using YOLOv5 with a PySimpleGUI-based GUI. 
Supports live webcam detection, automatic video recording, detection logging with timestamps, playback of recorded videos, and deletion of videos.

## Features
- Live detection from webcam with bounding boxes
- Automatic video recording of detection sessions
- Logs detection events with timestamps
- User-friendly GUI to start detection, play recorded videos, view logs, delete videos, and refresh the list

## Tech Stack
- Python
- OpenCV
- PySimpleGUI
- YOLOv5 (PyTorch)

## How to Run

Run these commands in your terminal:

```bash
git clone https://github.com/Tejaswini2906/live-object-detection-yolov5.git
cd live-object-detection-yolov5
pip install -r requirements.txt
python yolo_gui.py
