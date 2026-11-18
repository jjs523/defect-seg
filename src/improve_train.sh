#!/bin/sh
# 논문 조건에서 누수만 제거(dataset/: 원본 겹침 0)하고 시드 3개로 60 epoch 학습한다. 끊겨도 다시 실행하면 이어서 한다.
cd /home/defect-seg || exit 1
PY=.venv/bin/python
one() {
  N=clean_s$1; L=improve_$1.log; W=runs/$N/weights
  [ -f runs/$N/train_time.json ] && return 0
  if [ -f $W/last.pt ]; then
    $PY -c "from ultralytics import YOLO; YOLO('$W/last.pt').train(resume=True)" >> $L 2>&1 && echo '{"note":"resumed"}' > runs/$N/train_time.json
  else
    $PY src/week2_train.py --data dataset/data.yaml --patience 0 --seed $1 --name $N >> $L 2>&1
  fi
  echo "seed $1 rc=$? $(date -u +%H:%M:%S)" >> improve_status.log
}
echo "start $(date -u +%H:%M:%S)" >> improve_status.log
while :; do sleep 60; echo hb; done &
HB=$!
one 42 & P1=$!
one 1 & P2=$!
one 2 & P3=$!
wait $P1 $P2 $P3
kill $HB
echo "ALL_TRAIN_DONE $(date -u +%H:%M:%S)" >> improve_status.log
