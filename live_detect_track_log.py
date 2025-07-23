import cv2
import torch
import os
from datetime import datetime
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import non_max_suppression, check_img_size, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# Setup
weights = 'yolov5s.pt'
data = 'data/coco128.yaml'
device = select_device('')
imgsz = [640, 640]
conf_thres = 0.25
iou_thres = 0.45

model = DetectMultiBackend(weights, device=device, data=data)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

# Load webcam
dataset = LoadStreams(str(0), img_size=imgsz, stride=stride, auto=pt)

# Setup logging directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
img_dir = f"runs/detect_log/images"
vid_dir = f"runs/detect_log/videos"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
img_path = os.path.join(img_dir, f"{timestamp}.jpg")
vid_path = os.path.join(vid_dir, f"{timestamp}.mp4")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device).float() / 255.0
    if im.ndimension() == 3:
        im = im[None]

    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):
        im0 = im0s[i].copy()
        if out is None:
            h, w = im0.shape[:2]
            out = cv2.VideoWriter(vid_path, fourcc, 20, (w, h))

        annotator = Annotator(im0, line_width=2, example=str(names))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))

        result_frame = annotator.result()
        out.write(result_frame)
        cv2.imshow("Detection", result_frame)
        cv2.imwrite(img_path, result_frame)  # save only once
        if cv2.waitKey(1) == ord('q'):
            break

out.release()
cv2.destroyAllWindows()
print(f"✅ Saved: {img_path} and {vid_path}")
