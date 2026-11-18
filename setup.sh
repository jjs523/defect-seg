#!/bin/bash
# Step 3 · 1주차 단계 1 — 실습 환경 구축 (강의 자료 9쪽)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
grep -q "unset PYTHONPATH" .venv/bin/activate || sed -i '1a unset PYTHONPATH' .venv/bin/activate   # ROS 등 시스템 패키지 차단
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.lock.txt
python -c "import torch; from ultralytics import YOLO; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); YOLO('yolov8s-seg.pt'); print('model loaded')"
