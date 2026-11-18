"""데이터 구성 감사: 개수, 해상도, 클래스, 증강/원본, 분할 간 중복."""
import collections, csv, glob, hashlib, json, os, re
import cv2

ROOT = "data"
NAMES = ["crack", "efflorescence", "spall_detach"]
OUT = "results/week1"
os.makedirs(OUT, exist_ok=True)


def origin(stem):
    """파일명에서 원본 프레임 이름을 복원한다."""
    m = re.match(r"^(U[A-Z]{2}\d+_\d{4}_\d{2}_\d{2}_\d+_\d+)", stem)
    return m.group(1) if m else stem


rows, summary, unreadable = [], {}, []
hashes = collections.defaultdict(list)
for split in ["train", "val", "test"]:
    files = sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg"))
    res, cls_img, cls_inst, kinds = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    combos, areas, pts, bad = collections.Counter(), collections.defaultdict(list), [], 0
    empty = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        hashes[hashlib.md5(open(f, "rb").read()).hexdigest()].append((split, stem))
        im = cv2.imread(f)
        if im is None:  # 0바이트 등 읽을 수 없는 이미지는 집계에서 빼고 따로 기록한다
            unreadable.append(dict(split=split, stem=stem, size=os.path.getsize(f), origin=origin(stem)))
            continue
        h, w = im.shape[:2]
        res[f"{w}x{h}"] += 1
        kind = "roboflow" if ".rf." in stem else ("plain" if origin(stem) == stem else "aug")
        kinds[kind] += 1
        present = set()
        lab = f"{ROOT}/labels/{split}/{stem}.txt"
        lines = open(lab).read().strip().splitlines() if os.path.exists(lab) else []
        if not lines:
            empty += 1
        for ln in lines:
            v = ln.split()
            c = int(v[0])
            xy = list(map(float, v[1:]))
            if len(xy) < 6 or len(xy) % 2 or c not in (0, 1, 2) or min(xy) < 0 or max(xy) > 1:
                bad += 1
                continue
            present.add(c)
            cls_inst[c] += 1
            x, y = xy[0::2], xy[1::2]
            a = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i] for i in range(len(x))))
            areas[c].append(a)
        for c in present:
            cls_img[c] += 1
        combos["+".join(NAMES[c] for c in sorted(present)) or "none"] += 1
        rows.append(dict(split=split, stem=stem, origin=origin(stem), kind=kind, w=w, h=h,
                         crack=int(0 in present), efflorescence=int(1 in present), spall_detach=int(2 in present)))
    med = {NAMES[c]: round(sorted(a)[len(a) // 2] * 100, 3) for c, a in areas.items()}
    summary[split] = dict(images=len(files), resolutions=res.most_common(8), kinds=dict(kinds), empty_labels=empty,
                          bad_polygons=bad, images_per_class={NAMES[c]: cls_img[c] for c in range(3)},
                          instances_per_class={NAMES[c]: cls_inst[c] for c in range(3)},
                          median_area_pct=med, combos=combos.most_common())

orig = {s: {r["origin"] for r in rows if r["split"] == s} for s in ["train", "val", "test"]}
leak = {"train&val": sorted(orig["train"] & orig["val"]), "train&test": sorted(orig["train"] & orig["test"]),
        "val&test": sorted(orig["val"] & orig["test"])}
dups = [v for v in hashes.values() if len({s for s, _ in v}) > 1]
summary["unique_origins"] = {s: len(v) for s, v in orig.items()}
summary["origin_overlap"] = {k: len(v) for k, v in leak.items()}
summary["origin_overlap_examples"] = {k: v[:8] for k, v in leak.items()}
summary["identical_bytes_across_splits"] = len(dups)
summary["same_bytes_within_any"] = sum(len(v) > 1 for v in hashes.values())

with open(f"{OUT}/data_index.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summary["unreadable"] = dict(count=len(unreadable), by_origin=collections.Counter(u["origin"] for u in unreadable).most_common(),
                             sizes=sorted({u["size"] for u in unreadable}))
with open(f"{OUT}/unreadable_images.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["split", "stem", "size", "origin"])
    w.writeheader()
    w.writerows(unreadable)
json.dump(summary, open(f"{OUT}/data_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
