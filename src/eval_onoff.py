"""mask mAP와 영상 단위 ON/OFF 평가.
--stage val : 검증셋에서 클래스별 임계값을 정해 thresholds.json에 고정한다.
--stage test: 고정된 임계값을 그대로 테스트셋에 한 번 적용한다. 임계값이 없거나 이미 결과가 있으면 멈춘다."""
import argparse, csv, datetime, glob, hashlib, json, os, sys
import yaml
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES, origin_id

THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]  # 논문 Table 3, 강의 3주차
CONF_FLOOR = 0.001  # predict 기본값 0.25를 쓰면 그 아래 임계값이 모두 같은 결과가 된다

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["val", "test"], required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
THR_FILE, DONE = f"{a.out}/thresholds.json", f"{a.out}/{a.stage}_metrics.json"
if a.stage == "test":
    assert os.path.exists(THR_FILE), "검증셋 임계값이 먼저 있어야 한다"
    assert not os.path.exists(DONE), "테스트 평가는 이미 끝났다. 다시 돌리지 않는다"

cfg = yaml.safe_load(open(a.data))
root = cfg["path"]
model = YOLO(a.weights)


def score_images(split):
    """영상별로 정답 클래스와 클래스별 최고 confidence를 구한다."""
    rows = []
    for p in sorted(glob.glob(f"{root}/images/{split}/*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(f"{root}/labels/{split}/{stem}.txt") as fp:
            gt = {int(l.split()[0]) for l in fp if l.strip()}
        r = model.predict(p, imgsz=960, conf=CONF_FLOOR, verbose=False, device=0)[0]
        best = [0.0, 0.0, 0.0]
        for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            best[c] = max(best[c], s)
        rows.append(dict(stem=stem, gt=sorted(gt), conf=best))
    return rows


def onoff(rows, cid, thr):
    tp = fp = fn = tn = 0
    for r in rows:
        g, q = cid in r["gt"], r["conf"][cid] >= thr
        tp += g and q
        fp += (not g) and q
        fn += g and not q
        tn += not g and not q
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=f1)


def summarize(rows, thr):
    per = {NAMES[c]: onoff(rows, c, thr[NAMES[c]]) for c in range(3)}
    per["macro"] = {k: sum(per[NAMES[c]][k] for c in range(3)) / 3 for k in ("precision", "recall", "f1")}
    return per


def mask_map(split):
    m = model.val(data=os.path.abspath(a.data), split=split, imgsz=960, plots=True, device=0,
                  project=os.path.abspath(a.out), name=f"{split}_map", exist_ok=True, verbose=False)
    return dict(mask_map50=float(m.seg.map50), mask_map50_95=float(m.seg.map), box_map50=float(m.box.map50),
                per_class_mask_map50={NAMES[int(c)]: float(m.seg.ap50[i]) for i, c in enumerate(m.seg.ap_class_index)},
                per_class_mask_map50_95={NAMES[int(c)]: float(m.seg.ap[i]) for i, c in enumerate(m.seg.ap_class_index)})


def write_images(rows, thr, path):
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["stem", "gt", *[f"conf_{n}" for n in NAMES], *[f"on_{n}" for n in NAMES]])
        for r in rows:
            w.writerow([r["stem"], ";".join(NAMES[c] for c in r["gt"]), *[round(s, 5) for s in r["conf"]],
                        *[int(r["conf"][c] >= thr[NAMES[c]]) for c in range(3)]])


rows = score_images(a.stage)
if a.stage == "val":
    thr, curve = {}, []
    for c, n in enumerate(NAMES):
        scores = [onoff(rows, c, t) for t in THRS]
        curve += [dict(cls=n, **s) for s in scores]
        top = max(s["f1"] for s in scores)
        thr[n] = next(s["thr"] for s in scores if s["f1"] == top)  # 동점이면 낮은 임계값
    with open(f"{a.out}/val_threshold_search.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    json.dump(dict(thresholds=thr, weights_sha256=hashlib.sha256(open(a.weights, "rb").read()).hexdigest(),
                   fixed_at_utc=datetime.datetime.utcnow().isoformat()), open(THR_FILE, "w"), indent=1)
else:
    thr = json.load(open(THR_FILE))["thresholds"]

result = dict(split=a.stage, thresholds=thr, onoff=summarize(rows, thr), **mask_map(a.stage))
if a.stage == "test":
    train_origins = {origin_id(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f"{root}/images/train/*.jpg")}
    kept = [r for r in rows if origin_id(r["stem"]) not in train_origins]
    result["origin_overlap_images"] = [r["stem"] for r in rows if origin_id(r["stem"]) in train_origins]
    result["onoff_without_overlap"] = summarize(kept, thr)
write_images(rows, thr, f"{a.out}/{a.stage}_images.csv")
json.dump(result, open(DONE, "w"), indent=1, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False))
