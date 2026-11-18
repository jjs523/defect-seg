"""제출물 ②③⑥의 근거 수치: 강의 코드로 포맷·무결성 재확인, 원본 ID 방식 비교, 균열 폭 추정, best.pt 재검증(val만)."""
import argparse, collections, glob, json, os, re, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES, origin_id

ap = argparse.ArgumentParser(); ap.add_argument("--revalidate", action="store_true"); a = ap.parse_args()
OUT = Path("results/week2")

# 강의 10쪽 포맷 확인(같은 정렬·같은 판정)
lbls = sorted(glob.glob("dataset/labels/**/*.txt", recursive=True))
line = open(lbls[0], encoding="utf-8").readline().split()
fmt = "Detection" if len(line) == 5 else "Segmentation" if len(line) >= 7 and (len(line) - 1) % 2 == 0 else "확인 필요"
txt = f"라벨 파일 수: {len(lbls)}\n첫 라벨 파일: {Path(lbls[0]).name}\n토큰 개수: {len(line)}\n앞부분: {line[:9]}\n{fmt} 포맷입니다.\n"
# 강의 11쪽 무결성 점검
bad = set()
for p in lbls:
    for ln in open(p):
        v = ln.split()
        if not v:
            continue
        if int(v[0]) not in (0, 1, 2) or any(not 0 <= float(x) <= 1 for x in v[1:]):
            bad.add(p)
txt += f"이상 파일: {len(bad)}\n"
(OUT / "format_check.txt").write_text(txt); print(txt)

stems = {s: [Path(p).stem for p in glob.glob(f"dataset/images/{s}/*.jpg")] for s in ["train", "val", "test"]}
raw = {s: [Path(p).stem for p in glob.glob(f"data/images/{s}/*.jpg")] for s in ["train", "val", "test"]}
pat = re.compile(r"^U[A-Z]{2}\d+_\d{4}_\d{2}_\d{2}_\d+_\d+")
lecture = lambda s: s.split("_aug")[0]
held = {f(x): None for f in [origin_id] for x in raw["val"] + raw["test"]}
held_lecture = {lecture(x) for x in raw["val"] + raw["test"]}
video = lambda s: origin_id(s).rsplit("_", 1)[0]
vids = {k: {video(x) for x in v} for k, v in stems.items()}
chk = dict(
    id_pattern_miss={k: sum(not pat.match(x) for x in v) for k, v in raw.items()},
    aug_named={k: sum("_aug" in x or "flip" in x or "rot" in x for x in v) for k, v in stems.items()},
    train_overlap_files_our_id=sum(origin_id(x) in held for x in raw["train"]),
    train_overlap_files_lecture_id=sum(lecture(x) in held_lecture for x in raw["train"]),
    frame_overlap_after={f"train&{k}": len({origin_id(x) for x in stems["train"]} & {origin_id(x) for x in stems[k]}) for k in ["val", "test"]},
    shared_video={k: dict(shared=len(vids["train"] & vids[k]), total=len(vids[k])) for k in ["val", "test"]},
    non_jpg_files={k: sum(1 for p in Path(f"data/images/{k}").iterdir() if p.suffix.lower() != ".jpg") for k in raw},
)
# 균열 폭 추정: 띠 모양 폴리곤의 폭 ≈ 2 × 면적 / 둘레. 원본 프레임의 제공 라벨만 사용
widths, crack_frames = [], set()
for s in stems["train"]:
    if origin_id(s) != s:
        continue
    im = cv2.imread(f"dataset/images/train/{s}.jpg"); H, W = im.shape[:2]
    for l in open(f"dataset/labels/train/{s}.txt"):
        v = l.split()
        if v and int(v[0]) == 0:
            xy = (np.array(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.float32)
            area, per = cv2.contourArea(xy), cv2.arcLength(xy, True)
            if per > 0:
                crack_frames.add(s)
                widths.append(2 * area / per / max(H, W))   # 긴 변 대비 비율
w = np.array(widths)
chk["crack_width"] = dict(instances=len(w), frames=len(crack_frames), **{f"px_at_{n}": {f"q{q}": round(float(np.percentile(w, q) * n), 1) for q in (10, 50, 90)} for n in (1920, 960, 640)})
print(json.dumps(chk, ensure_ascii=False, indent=1))
json.dump(chk, open(OUT / "split_checks.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if a.revalidate:
    from ultralytics import YOLO
    m = YOLO("results/week2/train/weights/best.pt")
    r = m.val(data=os.path.abspath("dataset/data.yaml"), split="val", imgsz=960, batch=4, device=0, plots=False, verbose=False, project=os.path.abspath("runs"), name="week2_val_check", exist_ok=True)
    idx = list(r.seg.ap_class_index)
    res = dict(split="val", images=40, mask_map50=round(float(r.seg.map50), 4), mask_map=round(float(r.seg.map), 4),
               per_class={NAMES[c]: dict(map50=round(float(r.seg.ap50[i]), 4), map=round(float(r.seg.ap[i]), 4)) for i, c in enumerate(idx)})
    print(res)
    json.dump(res, open(OUT / "best_pt_val_check.json", "w"), indent=1)
