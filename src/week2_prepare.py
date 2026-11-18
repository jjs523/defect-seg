"""제공 데이터(data/)에서 학습용 dataset/을 만든다.
val·test는 그대로 두고, train에서 (1) 읽을 수 없는 영상 (2) val·test와 원본이 같은 영상을 뺀다.
영상과 라벨 대응, 클래스 ID, 좌표 범위, 폴리곤 점 수를 함께 점검한다."""
import collections, csv, json, os, shutil, sys
from pathlib import Path
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES, origin_id

SRC, DST, OUT = Path("data"), Path("dataset"), Path("results/week2")
if not any((SRC / "images" / "train").glob("*.jpg")):
    sys.exit("data/images/train에 영상이 없습니다. 제공 데이터를 data/에 풀어 주세요.")
OUT.mkdir(parents=True, exist_ok=True)
if DST.exists():
    shutil.rmtree(DST)

held = {origin_id(p.stem) for s in ["val", "test"] for p in (SRC / "images" / s).glob("*.jpg")}
excluded, report, seen = [], {}, collections.defaultdict(set)
for split in ["train", "val", "test"]:
    (DST / "images" / split).mkdir(parents=True)
    (DST / "labels" / split).mkdir(parents=True)
    imgs = sorted((SRC / "images" / split).glob("*.jpg"))
    r = collections.Counter(files=len(imgs))
    inst, per_img = collections.Counter(), collections.Counter()
    lab_stems = {p.stem for p in (SRC / "labels" / split).glob("*.txt")}
    r["label_without_image"] = len(lab_stems - {p.stem for p in imgs})
    for p in imgs:
        lab = SRC / "labels" / split / f"{p.stem}.txt"
        if not lab.exists():
            r["image_without_label"] += 1
            continue
        reason = None
        if p.stat().st_size == 0 or cv2.imread(str(p)) is None:
            reason = "unreadable"
        elif split == "train" and origin_id(p.stem) in held:
            reason = "origin_in_val_or_test"
        if reason:
            r[reason] += 1
            excluded.append(dict(split=split, stem=p.stem, origin=origin_id(p.stem), reason=reason))
            continue
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            r["empty_label"] += 1
        for v in lines:
            xy = list(map(float, v[1:]))
            if int(v[0]) not in (0, 1, 2):
                r["bad_class_id"] += 1
            if not all(0 <= x <= 1 for x in xy):
                r["bad_coord"] += 1
            if len(xy) < 6 or len(xy) % 2:
                r["bad_polygon"] += 1
            inst[NAMES[int(v[0])]] += 1
        for c in {int(v[0]) for v in lines}:
            per_img[NAMES[c]] += 1
        os.link(p, DST / "images" / split / p.name)
        os.link(lab, DST / "labels" / split / lab.name)
        r["kept"] += 1
        seen[split].add(origin_id(p.stem))
    report[split] = dict(r, images_per_class=dict(per_img), instances_per_class=dict(inst), origins=len(seen[split]))

report["origin_overlap_after"] = {f"{a}&{b}": len(seen[a] & seen[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
assert not any(report["origin_overlap_after"].values()), "분할 간 원본 겹침이 남아 있음"
(DST / "data.yaml").write_text(f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crack\n  1: efflorescence\n  2: spall_detach\n")
with open(OUT / "excluded_images.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["split", "stem", "origin", "reason"])
    w.writeheader()
    w.writerows(excluded)
json.dump(report, open(OUT / "integrity.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, ensure_ascii=False))
first = sorted((DST / "labels" / "train").glob("*.txt"))[0]
tok = first.read_text().splitlines()[0].split()
fmt = "Segmentation" if len(tok) >= 7 and (len(tok) - 1) % 2 == 0 else "확인 필요"
txt = f"라벨 파일: {first.name}\n토큰 개수: {len(tok)}\n앞부분: {tok[:9]}\n판정: {fmt} 포맷\n"
(OUT / "format_check.txt").write_text(txt)
print(txt)
