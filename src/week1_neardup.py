"""분할 간 근접 중복 검사: 저해상도 정규화 상호상관(NCC), 뒤집기·180도 회전 포함. 숫자만 쓰고 test 영상을 관찰하지 않는다."""
import csv, glob, json, os
import cv2
import numpy as np


def vec(g):
    v = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6) / np.sqrt(v.size)


items = {}
for split in ["train", "val", "test"]:
    items[split] = []
    for f in sorted(glob.glob(f"data/images/{split}/*.jpg")):
        g = cv2.imread(f, 0)
        if g is None:
            continue
        items[split].append((os.path.basename(f)[:-4], [vec(x) for x in (g, g[:, ::-1], g[::-1, :], g[::-1, ::-1])]))
rows = []
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    A = np.stack([v for _, vs in items[a] for v in vs])
    for stem, vs in items[b]:
        c = A @ vs[0]
        k = int(c.argmax())
        rows.append(dict(a=a, b=b, b_stem=stem, nearest_a=items[a][k // 4][0], orient=["same", "hflip", "vflip", "rot180"][k % 4], ncc=round(float(c[k]), 4)))
rows.sort(key=lambda r: -r["ncc"])
with open("results/week1/near_duplicates.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
out = {}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    x = [r["ncc"] for r in rows if r["a"] == a and r["b"] == b]
    out[f"{b} vs {a}"] = {"n": len(x), "ncc>=0.95": sum(v >= .95 for v in x), "ncc>=0.90": sum(v >= .9 for v in x), "ncc>=0.80": sum(v >= .8 for v in x), "max": max(x), "median": round(float(np.median(x)), 3)}
json.dump(out, open("results/week1/near_duplicates_summary.json", "w"), indent=1)
print(json.dumps(out))
for r in rows[:16]:
    print(r["ncc"], r["a"], r["b"], r["orient"], r["b_stem"][:40], "<-", r["nearest_a"][:52])
