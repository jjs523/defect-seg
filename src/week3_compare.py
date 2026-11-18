"""모델별 성능 비교와 혼동행렬(README용). 테스트를 다시 돌리지 않고 저장된 영상별 최고 confidence와 검증셋 임계값으로 계산한다.
모델: clean 시드 42·1·2(누수 제거, 1이 제출 모델), paper(논문 조건 재현, 누수 포함).
혼동행렬은 영상 단위다. 테스트 영상마다 정답 클래스가 하나이므로 행=정답 클래스, 열=예측 클래스(임계값을 넘은 클래스 중 confidence 최고, 없으면 '없음').
클래스별 Precision·Recall·F1은 강의의 ON/OFF 정의(클래스마다 따로 판정)를 따른다."""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES

OUT = "results/week3"
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(), "axes.unicode_minus": False})
done = json.load(open("results/improve/test_results.json", encoding="utf-8"))
paper = json.load(open("results/paper/test_metrics.json", encoding="utf-8"))


def rows_improve(seed):
    return [dict(stem=r["stem"], gt={NAMES.index(g) for g in r["gt"].split(";") if g}, conf=[float(r[f"plain_{n}"]) for n in NAMES])
            for r in csv.DictReader(open(f"results/improve/test_conf_s{seed}.csv", encoding="utf-8"))]


def rows_paper():
    return [dict(stem=r["stem"], gt={NAMES.index(g) for g in r["gt"].split(";") if g}, conf=[float(r[f"conf_{n}"]) for n in NAMES])
            for r in csv.DictReader(open("results/paper/test_images.csv", encoding="utf-8"))]


MODELS = [(f"clean_s{s}" + (" (제출)" if s == done["selected_seed"] else ""), rows_improve(s), done["seeds"][s]["thresholds"],
           done["seeds"][s]["mask_map50"], done["seeds"][s]["mask_map50_95"]) for s in ("42", "1", "2")]
MODELS.append(("paper_repro (누수 포함)", rows_paper(), paper["thresholds"], paper["mask_map50"], paper["mask_map50_95"]))

table, mats = [], {}
for name, rows, thr, m50, m5095 in MODELS:
    per = {}
    for c, n in enumerate(NAMES):
        tp = sum(1 for r in rows if c in r["gt"] and r["conf"][c] >= thr[n]); fp = sum(1 for r in rows if c not in r["gt"] and r["conf"][c] >= thr[n])
        fn = sum(1 for r in rows if c in r["gt"] and r["conf"][c] < thr[n]); tn = len(rows) - tp - fp - fn
        per[n] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0,
                      f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    m = np.zeros((3, 4), int)
    for r in rows:
        g = next(iter(r["gt"]))
        on = [c for c in range(3) if r["conf"][c] >= thr[NAMES[c]]]
        m[g, max(on, key=lambda c: r["conf"][c]) if on else 3] += 1
    mats[name] = m
    exact = sum(1 for r in rows if {c for c in range(3) if r["conf"][c] >= thr[NAMES[c]]} == r["gt"])
    table.append(dict(model=name, thresholds=thr, mask_map50=m50, mask_map50_95=m5095,
                      accuracy=float(np.trace(m[:, :3]) / m.sum()), exact_match=exact / len(rows),
                      precision=float(np.mean([per[n]["precision"] for n in NAMES])), recall=float(np.mean([per[n]["recall"] for n in NAMES])),
                      f1=float(np.mean([per[n]["f1"] for n in NAMES])), per_class=per, confusion=m.tolist()))

json.dump(table, open(f"{OUT}/model_comparison.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(f"{OUT}/model_comparison.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["model", "mask_mAP50", "mask_mAP50-95", "accuracy", "macro_precision", "macro_recall", "macro_f1"])
    for t in table:
        w.writerow([t["model"], round(t["mask_map50"], 4), round(t["mask_map50_95"], 4), round(t["accuracy"], 4), round(t["precision"], 4), round(t["recall"], 4), round(t["f1"], 4)])

fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
for ax, t in zip(axs, table):
    m = np.array(t["confusion"])
    ax.imshow(m, cmap="Blues", vmin=0, vmax=14)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=13, color="white" if m[i, j] > 8 else "#111")
    ax.set_xticks(range(4)); ax.set_xticklabels(["crack", "efflor.", "spall", "없음"]); ax.set_yticks(range(3)); ax.set_yticklabels(["crack", "efflor.", "spall"])
    ax.set_xlabel("예측"); ax.set_title(f"{t['model']}\nAcc {t['accuracy']:.3f} · macro F1 {t['f1']:.3f}", fontsize=10.5)
axs[0].set_ylabel("정답")
fig.suptitle("테스트셋 40장 영상 단위 혼동행렬 (검증셋 임계값 고정 적용)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrices.png", dpi=150); plt.close(fig)
for t in table:
    print(f"{t['model']:<24} mAP50 {t['mask_map50']:.3f} mAP50-95 {t['mask_map50_95']:.3f} Acc {t['accuracy']:.3f} exact {t['exact_match']:.3f} "
          f"P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f}\n{np.array(t['confusion'])}")
