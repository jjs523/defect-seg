"""YOLOv8s-Seg 학습. 강의 13쪽 설정 그대로. 예: python src/week2_train.py --epochs 3 --name smoke"""
import argparse, json, os, time
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--name", default="step3")
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--patience", type=int, default=20, help="0이면 조기 종료 없이 끝까지 학습")
a = ap.parse_args()
t0 = time.time()
YOLO("yolov8s-seg.pt").train(
    data=os.path.abspath(a.data), epochs=a.epochs, imgsz=960, batch=a.batch, optimizer="AdamW", lr0=0.00015,
    mosaic=0.0, mixup=0.0, patience=a.patience, seed=a.seed, project=os.path.abspath("runs"), name=a.name, exist_ok=True, fraction=a.fraction, device=0, plots=True)
json.dump(dict(seconds=round(time.time() - t0), epochs=a.epochs, batch=a.batch), open(f"runs/{a.name}/train_time.json", "w"))
print("DONE", round(time.time() - t0), "s")
