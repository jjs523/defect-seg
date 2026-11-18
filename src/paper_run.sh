#!/bin/sh
# 논문 조건 재현: 학습(60 epoch, 조기 종료 없음) -> 검증셋 임계값 고정 -> 테스트 1회. 끊겨도 다시 실행하면 이어서 한다.
cd /home/defect-seg || exit 1
PY=.venv/bin/python; LOG=paper.log; W=runs/paper/weights
step() { echo "=== $1 $(date -u +%H:%M:%S)" >> $LOG; shift; "$@" >> $LOG 2>&1 || { echo "FAILED: $*" >> $LOG; exit 1; }; }
step prepare $PY src/paper_prepare.py
if [ ! -f runs/paper/train_time.json ]; then
  if [ -f $W/last.pt ]; then
    step resume $PY -c "from ultralytics import YOLO; YOLO('$W/last.pt').train(resume=True)"
    echo '{"note":"resumed; see results.csv time column"}' > runs/paper/train_time.json
  else
    step train $PY src/week2_train.py --data dataset_paper/data.yaml --patience 0 --name paper
  fi
fi
[ -f results/paper/thresholds.json ] || step val $PY src/eval_onoff.py --stage val --weights $W/best.pt --data dataset_paper/data.yaml --out results/paper
[ -f results/paper/test_metrics.json ] || step test $PY src/eval_onoff.py --stage test --weights $W/best.pt --data dataset_paper/data.yaml --out results/paper
echo "ALL_DONE $(date -u +%H:%M:%S)" >> $LOG
