"""1주차 그림: samples9, 하드 네거티브, 클래스 예시. 관찰 대상은 train 원본뿐이고 test는 보지 않는다."""
import csv, json, random
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/week1"
NAMES = ["crack", "efflorescence", "spall_detach"]
COL = {0: (255, 220, 0), 1: (0, 230, 230), 2: (255, 120, 0)}
cfg = json.load(open("src/week1_examples.json"))
rows = [r for r in csv.DictReader(open(f"{OUT}/data_index.csv")) if r["split"] == "train" and r["kind"] == "plain"]


def load(stem, outline=True):
    im = cv2.cvtColor(cv2.imread(f"data/images/train/{stem}.jpg"), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    if outline:
        for ln in open(f"data/labels/train/{stem}.txt"):
            v = ln.split()
            xy = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.polylines(im, [xy], True, COL[int(v[0])], 5)
    return im


# samples9: 강의 코드와 같은 3x3 격자. 클래스별 3장을 고정 시드로 뽑고, 윤곽선은 넣지 않는다.
random.seed(42)
sample = [(n, r["stem"]) for n in NAMES for r in random.sample([r for r in rows if r[n] == "1"], 3)]
fig, ax = plt.subplots(3, 3, figsize=(10, 9))
for a, (n, stem) in zip(ax.ravel(), sample):
    a.imshow(load(stem, False))
    a.set_title(f"{n} | {stem[:24]}", fontsize=7)
    a.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/samples9.png", dpi=110)
plt.close()

# 밝기 통계: train 원본 전체
st = []
for r in rows:
    g = cv2.imread(f"data/images/train/{r['stem']}.jpg", 0)
    st.append((g.mean(), g.std()))
st = np.array(st)
b = {"n": len(rows), "mean_q10_50_90": np.percentile(st[:, 0], [10, 50, 90]).round(1).tolist(), "mean_min_max": [round(float(st[:, 0].min()), 1), round(float(st[:, 0].max()), 1)],
     "std_q10_50_90": np.percentile(st[:, 1], [10, 50, 90]).round(1).tolist()}
json.dump(b, open(f"{OUT}/brightness.json", "w"), indent=1)
print(b)

# 하드 네거티브: 라벨 윤곽선은 색, 혼동 요소는 빨간 상자
hn = cfg["hard_negatives"]
fig, ax = plt.subplots(3, 3, figsize=(12, 7.4))
ax = ax.ravel()
for a in ax:
    a.axis("off")
for a, h in zip(ax, hn):
    im = load(h["stem"])
    H, W = im.shape[:2]
    x0, y0, x1, y1 = h["box"]
    cv2.rectangle(im, (int(x0 * W) + 4, int(y0 * H) + 4), (int(x1 * W) - 4, int(y1 * H) - 4), (230, 30, 30), 8)
    a.imshow(im)
    a.set_title(f"{h['key']}  {h['stem'][:27]}", fontsize=8)
for a, (t, s) in zip(ax[len(hn):], [("repair labeled (contrast)", cfg["contrast"]["repair_labeled"]), ("unlabeled cracks", cfg["contrast"]["unlabeled_crack"])]):
    a.imshow(load(s))
    a.set_title(f"{t}  {s[:24]}", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/hard_negatives.png", dpi=130)
plt.close()
for h in hn:
    im = load(h["stem"], False)
    H, W = im.shape[:2]
    x0, y0, x1, y1 = h["box"]
    cv2.imwrite(f"{OUT}/hn_{h['key']}.jpg", cv2.cvtColor(im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

# 기준서용 클래스 예시
fig, ax = plt.subplots(1, 3, figsize=(12, 2.5))
for a, c in zip(ax, cfg["classes"]):
    a.imshow(load(c["stem"]))
    a.set_title(c["cls"], fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/class_examples.png", dpi=130)
plt.close()
