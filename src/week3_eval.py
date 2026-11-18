"""3주차 평가 정리. 제출 모델은 results/week3/model_decision.json의 clean_s1이다.
임계값과 모델은 2026-11-05 검증셋으로 고정한 results/improve/protocol.json을 그대로 쓰며 이 파일에는 값을 고르는 코드가 없다.
1) 검증셋: 저장된 영상별 최고 confidence(val_conf_s1.csv)로 클래스별 F1 곡선을 다시 그리고, 고른 값이 protocol과 같은지 확인한다.
2) 테스트 ON/OFF: test_conf_s1.csv에 고정 임계값을 적용하고 11/5 결과와 같은지 확인한다.
3) 보고용 테스트 실행(--report-pass): 클래스별 mask mAP50-95와 FP/FN 그림을 위해 같은 모델로 한 번 더 추론한다.
   임계값·모델은 바꾸지 않으며 실행 사실을 test_attempts.log에 남긴다."""
import argparse, csv, datetime, glob, hashlib, json, os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES

IMP, OUT = "results/improve", "results/week3"
SEED = "1"
WEIGHTS = f"{IMP}/train_s{SEED}/weights/best.pt"
LECTURE_GRID = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]
GRID = LECTURE_GRID + [0.4, 0.5, 0.6, 0.7]
COL = {"crack": "#2a6fdb", "efflorescence": "#c2410c", "spall_detach": "#15803d"}
BGR = {0: (40, 40, 230), 1: (230, 120, 30), 2: (40, 170, 40)}

ap = argparse.ArgumentParser()
ap.add_argument("--report-pass", action="store_true", help="보고용 테스트 추론(클래스별 mAP50-95, FP/FN 그림)")
ap.add_argument("--preds-only", action="store_true", help="오류 사례 영상의 예측만 다시 저장(그림용)")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
import yaml
ROOT_DS = yaml.safe_load(open("dataset/data.yaml"))["path"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def onoff(rows, cid, thr):
    tp = fp = fn = tn = 0
    for r in rows:
        g, q = cid in r["gt"], r["conf"][cid] >= thr
        tp += g and q; fp += (not g) and q; fn += g and not q; tn += (not g) and not q
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def load_conf(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append(dict(stem=r["stem"], gt={NAMES.index(g) for g in r["gt"].split(";") if g},
                         conf=[float(r[f"plain_{n}"]) for n in NAMES]))
    return rows


# --- 0. 고정값 확인 ---
proto = json.load(open(f"{IMP}/protocol.json", encoding="utf-8"))
done = json.load(open(f"{IMP}/test_results.json", encoding="utf-8"))
assert sha(f"{IMP}/protocol.json") == done["protocol_sha256"], "protocol.json이 테스트 때와 다르다"
assert done["selected_seed"] == SEED and done["selected_mode"] == "plain", "제출 모델이 protocol 선택과 다르다"
assert sha(WEIGHTS) == proto["seeds"][SEED]["weights_sha256"], "가중치가 protocol에 기록된 것과 다르다"
THR = proto["seeds"][SEED]["thresholds"]["plain"]

# --- 1. 검증셋 F1 곡선 ---
val = load_conf(f"{IMP}/val_conf_s{SEED}.csv")
curve, chosen = [], {}
for c, n in enumerate(NAMES):
    sc = [onoff(val, c, t) for t in GRID]
    for s in sc:
        curve.append(dict(cls=n, **s))
    f1 = [s["f1"] for s in sc]
    chosen[n] = GRID[f1.index(max(f1))]                       # 동점이면 낮은 값(강의 11쪽)
    lec = [s["f1"] for s in sc[:len(LECTURE_GRID)]]
    assert LECTURE_GRID[lec.index(max(lec))] == chosen[n], "강의 범위와 넓힌 범위의 선택이 다르다"
assert chosen == THR, f"다시 계산한 임계값 {chosen}이 protocol {THR}과 다르다"
with open(f"{OUT}/val_threshold_f1.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
val_tab = {n: onoff(val, c, THR[n]) for c, n in enumerate(NAMES)}

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (c, n) in zip(axs, enumerate(NAMES)):
    ys = [r["f1"] for r in curve if r["cls"] == n]
    ax.axvspan(LECTURE_GRID[-1] * 1.15, GRID[-1] * 1.3, color="#f1f1f1", lw=0)
    ax.plot(GRID, ys, "-o", color=COL[n], lw=2, ms=5)
    ax.axvline(THR[n], color="#444", ls="--", lw=1)
    ax.annotate(f"선택 {THR[n]}\nF1 {val_tab[n]['f1']:.3f}", (THR[n], val_tab[n]["f1"]), xytext=(10, -48),
                textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([0.001, 0.01, 0.1, 0.7]); ax.set_xticklabels(["0.001", "0.01", "0.1", "0.7"]); ax.minorticks_off()
    ax.set_title(n); ax.set_xlabel("confidence 임계값 (log)"); ax.grid(alpha=.3)
axs[0].set_ylabel("검증셋 영상 단위 F1"); axs[0].set_ylim(0.4, 1.02)
fig.text(0.99, 0.01, "회색: 강의 범위(0.001~0.3) 밖으로 넓힌 구간", ha="right", fontsize=8, color="#666")
fig.tight_layout(); fig.savefig(f"{OUT}/val_threshold_f1.png", dpi=160); plt.close(fig)

# --- 2. 테스트 ON/OFF (고정 임계값) ---
test = load_conf(f"{IMP}/test_conf_s{SEED}.csv")
tab = {n: onoff(test, c, THR[n]) for c, n in enumerate(NAMES)}
macro = sum(tab[n]["f1"] for n in NAMES) / 3
ref = done["seeds"][SEED]["onoff"]
for n in NAMES:
    assert all(tab[n][k] == ref[n][k] for k in ("tp", "fp", "fn", "tn")), f"{n} 표가 11/5 결과와 다르다"
with open(f"{OUT}/test_onoff.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["class", "threshold", "TP", "FP", "FN", "TN", "precision", "recall", "f1"])
    for n in NAMES:
        t = tab[n]; w.writerow([n, t["thr"], t["tp"], t["fp"], t["fn"], t["tn"], round(t["precision"], 4), round(t["recall"], 4), round(t["f1"], 4)])
    w.writerow(["macro F1", "", "", "", "", "", "", "", round(macro, 4)])

errors = []
for c, n in enumerate(NAMES):
    for r in test:
        g, q = c in r["gt"], r["conf"][c] >= THR[n]
        if g != q:
            errors.append(dict(cls=n, kind="FP" if q else "FN", stem=r["stem"], conf=r["conf"][c], thr=THR[n],
                               gt=";".join(NAMES[x] for x in sorted(r["gt"]))))
# crack은 오류가 없어 경계에 가장 가까웠던 사례를 함께 본다
c = 0
off = max((r for r in test if c not in r["gt"]), key=lambda r: r["conf"][c])
on = min((r for r in test if c in r["gt"]), key=lambda r: r["conf"][c])
near = [dict(cls="crack", kind="근접_OFF", stem=off["stem"], conf=off["conf"][c], thr=THR["crack"], gt=";".join(NAMES[x] for x in sorted(off["gt"]))),
        dict(cls="crack", kind="근접_ON", stem=on["stem"], conf=on["conf"][c], thr=THR["crack"], gt=";".join(NAMES[x] for x in sorted(on["gt"])))]
with open(f"{OUT}/error_cases.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=["cls", "kind", "stem", "conf", "thr", "gt"]); w.writeheader(); w.writerows(errors + near)

summary = dict(model=WEIGHTS, weights_sha256=sha(WEIGHTS), protocol_sha256=done["protocol_sha256"], thresholds=THR,
               val_onoff=val_tab, val_macro_f1=sum(val_tab[n]["f1"] for n in NAMES) / 3,
               test_onoff=tab, test_macro_f1=macro, test_mask_map50=done["seeds"][SEED]["mask_map50"],
               test_mask_map50_95=done["seeds"][SEED]["mask_map50_95"], test_per_class_mask_map50=done["seeds"][SEED]["per_class_mask_map50"],
               errors=errors, near=near, paper=done["paper"])

# --- 3. 보고용 테스트 실행 ---
PRED = f"{OUT}/error_preds.json"   # 오류 사례 영상의 예측(클래스, 점수, 폴리곤). 있으면 그림을 추론 없이 다시 그린다
if a.report_pass or a.preds_only:
    from ultralytics import YOLO
    with open(f"{IMP}/test_attempts.log", "a", encoding="utf-8") as fp:
        what = "class-wise mAP + error-case predictions" if a.report_pass else "error-case predictions only (figure redraw)"
        fp.write(f"{datetime.datetime.now().isoformat()} week3 report-only pass: {what} (clean_s1, thresholds fixed from protocol, no selection)\n")
    model = YOLO(WEIGHTS)
    if a.report_pass:
        m = model.val(data=os.path.abspath("dataset/data.yaml"), split="test", imgsz=960, plots=False, device=0,
                      project=os.path.abspath(f"{OUT}/val_runs"), name="test_report", exist_ok=True, verbose=False)
        per = {NAMES[int(cc)]: dict(map50=float(m.seg.ap50[i]), map50_95=float(m.seg.ap[i])) for i, cc in enumerate(m.seg.ap_class_index)}
        rerun = dict(mask_map50=float(m.seg.map50), mask_map50_95=float(m.seg.map), per_class=per)
        rerun["diff_vs_1105"] = dict(mask_map50=rerun["mask_map50"] - summary["test_mask_map50"],
                                     mask_map50_95=rerun["mask_map50_95"] - summary["test_mask_map50_95"])
        summary["test_rerun"] = rerun
    preds = {}
    for e in errors + near:
        if e["stem"] in preds:
            continue
        r = model.predict(f"{ROOT_DS}/images/test/{e['stem']}.jpg", imgsz=960, conf=0.001, verbose=False, device=0)[0]
        preds[e["stem"]] = [] if r.masks is None else [
            dict(cls=int(c), conf=float(q), xy=r.masks.xy[k].round().astype(int).tolist())
            for k, (c, q) in enumerate(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()))]
    json.dump(preds, open(PRED, "w"))

if os.path.exists(PRED):
    preds = json.load(open(PRED))
    os.makedirs(f"{OUT}/error_cases", exist_ok=True)
    for f in glob.glob(f"{OUT}/error_cases/*.jpg"):
        os.remove(f)
    for e in errors + near:
        img = cv2.imread(f"{ROOT_DS}/images/test/{e['stem']}.jpg"); H, W = img.shape[:2]
        cid = NAMES.index(e["cls"])
        mine = [q for q in preds[e["stem"]] if q["cls"] == cid]
        top = max([q["conf"] for q in mine], default=0.0)
        for q in mine:
            keep = q["conf"] >= e["thr"]
            xy = np.array(q["xy"], np.int32)
            if (keep or q["conf"] == top) and len(xy) >= 3:
                cv2.polylines(img, [xy], True, BGR[cid] if keep else (0, 200, 255), 4, cv2.LINE_AA)
                x0, y0 = xy.min(0)
                for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                    cv2.putText(img, f"{e['cls']} {q['conf']:.3f}", (int(x0), max(30, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, th)
        for l in open(f"{ROOT_DS}/labels/test/{e['stem']}.txt"):   # 정답은 분홍(분필 표시가 흰색이라 구분)
            v = l.split()
            if v:
                xy = (np.array(v[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
                cv2.polylines(img, [xy], True, (255, 0, 255), 5, cv2.LINE_AA)
        e["conf_rerun"] = top
        sc = 960 / max(H, W)
        cv2.imwrite(f"{OUT}/error_cases/{e['cls']}_{e['kind']}_{e['stem'][:40]}.jpg", cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary["errors"], summary["near"] = errors, near

if "test_rerun" in summary:
    pass
elif os.path.exists(f"{OUT}/week3_summary.json"):          # 테스트를 다시 돌리지 않을 때는 이전 보고용 실행 결과를 유지
    old = json.load(open(f"{OUT}/week3_summary.json", encoding="utf-8"))
    if "test_rerun" in old:
        summary["test_rerun"] = old["test_rerun"]
        prev = {(e["cls"], e["kind"], e["stem"]): e.get("conf_rerun") for e in old["errors"] + old["near"]}
        for e in errors + near:
            if prev.get((e["cls"], e["kind"], e["stem"])) is not None:
                e["conf_rerun"] = prev[(e["cls"], e["kind"], e["stem"])]
json.dump(summary, open(f"{OUT}/week3_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(dict(thresholds=THR, val_macro_f1=summary["val_macro_f1"], test_macro_f1=macro,
                      errors=[(e["cls"], e["kind"], e["stem"][:30], round(e["conf"], 4), round(e.get("conf_rerun", -1), 4)) for e in errors + near],
                      rerun=summary.get("test_rerun")), ensure_ascii=False, indent=1))
