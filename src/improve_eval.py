"""논문보다 나은지 판단하기 위한 평가. 강의 심화 과제의 시드 반복, 공통 임계값 대조, 타일 추론과 3주차 자료의 임계값 범위 조정을 적용한다.
--stage val : 검증셋만 써서 임계값, 추론 방식, 대표 시드를 정하고 protocol.json에 고정한다.
--stage test: protocol.json의 값을 그대로 테스트셋에 적용한다. 값을 바꾸는 코드가 없고, 결과가 있으면 다시 돌지 않는다.
--dry-on-val: 코드 점검용. test 단계 경로를 검증셋으로 돌리고 테스트셋은 읽지 않는다."""
import argparse, csv, datetime, glob, hashlib, json, os, statistics, sys
import cv2, yaml
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES, origin_id

LECTURE_GRID = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]
GRID = LECTURE_GRID + [0.4, 0.5, 0.6, 0.7]  # 3주차 16쪽: 분포를 보고 탐색 범위 조정
CONF_FLOOR, TILE = 0.001, 960
PAPER = dict(mask_map50=0.443, mask_map50_95=0.222, macro_f1=0.896)
MARGIN = 0.025  # 강의: 40장에서 1장이 2.5%p이고 그보다 작은 차이는 셀 수 없다

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["val", "test"], required=True)
ap.add_argument("--seeds", default="42,1,2")
ap.add_argument("--run-prefix", default="clean_s", help="runs/<prefix><seed>/weights/best.pt")
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--out", default="results/improve")
ap.add_argument("--dry-on-val", action="store_true")
ap.add_argument("--dry-force-tile", action="store_true", help="코드 점검용: 타일 경로도 시험한다")
ap.add_argument("--expect-epochs", type=int, default=60, help="0이면 완료 확인 생략(코드 점검용)")
ap.add_argument("--retry-reason", default="", help="테스트가 중간에 멈췄을 때만. 사유를 test_attempts.log에 남긴다")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
PROTO, DONE = f"{a.out}/protocol.json", f"{a.out}/test_results.json"
STARTED = f"{a.out}/test_started.json"
SEEDS = a.seeds.split(",")
cfg = yaml.safe_load(open(a.data))
ROOT = cfg["path"]


def weights(s):
    return f"runs/{a.run_prefix}{s}/weights/best.pt"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def video_id(stem):
    return origin_id(stem).rsplit("_", 1)[0]


def starts(length):
    """타일 시작 좌표. 절반씩 겹치고 마지막 타일은 끝에 맞춘다."""
    if length <= TILE:
        return [0]
    xs = list(range(0, length - TILE, TILE // 2))
    return xs + [length - TILE]


def score_split(model, split, want_tile):
    rows = []
    for p in sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(f"{ROOT}/labels/{split}/{stem}.txt") as fp:
            gt = {int(l.split()[0]) for l in fp if l.strip()}
        im = cv2.imread(p)

        def update(best, crop):
            r = model.predict(crop, imgsz=TILE, conf=CONF_FLOOR, verbose=False, device=0)[0]
            for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                best[c] = max(best[c], s)
        plain = [0.0, 0.0, 0.0]
        update(plain, im)
        tile = list(plain)  # 전체 영상 예측에 타일 예측을 더한다. 영상 단위 판정은 존재 여부만 보므로 타일 간 중복 제거가 필요 없다
        if want_tile:
            h, w = im.shape[:2]
            for y in starts(h):
                for x in starts(w):
                    update(tile, im[y:y + TILE, x:x + TILE])
        rows.append(dict(stem=stem, gt=sorted(gt), plain=plain, tile=tile))
    return rows


def onoff(rows, mode, cid, thr):
    tp = fp = fn = tn = 0
    for r in rows:
        g, q = cid in r["gt"], r[mode][cid] >= thr
        tp += g and q
        fp += (not g) and q
        fn += g and not q
        tn += not g and not q
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    # 분수 그대로 계산해야 같은 F1이 부동소수 오차 없이 동점이 된다
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def table(rows, mode, thr):
    per = {n: onoff(rows, mode, c, thr[n]) for c, n in enumerate(NAMES)}
    per["macro_f1"] = sum(per[n]["f1"] for n in NAMES) / 3
    return per


def per_class_thresholds(rows, mode, grid):
    out = {}
    for c, n in enumerate(NAMES):
        scores = [onoff(rows, mode, c, t)["f1"] for t in grid]
        out[n] = grid[scores.index(max(scores))]  # 동점이면 낮은 임계값
    return out


def common_threshold(rows, mode, grid):
    scores = [round(table(rows, mode, {n: t for n in NAMES})["macro_f1"], 9) for t in grid]
    return grid[scores.index(max(scores))]


def mask_map(model, split, name, data=None):
    m = model.val(data=os.path.abspath(data or a.data), split=split, imgsz=TILE, plots=False, device=0,
                  project=os.path.abspath(a.out), name=name, exist_ok=True, verbose=False)
    return dict(mask_map50=float(m.seg.map50), mask_map50_95=float(m.seg.map), box_map50=float(m.box.map50),
                per_class_mask_map50={NAMES[int(c)]: float(m.seg.ap50[i]) for i, c in enumerate(m.seg.ap_class_index)})


def present_macro_f1(rows, mode, thr):
    """부분 집합용: 정답이 있는 클래스만 평균한다."""
    t = table(rows, mode, thr)
    have = [n for n in NAMES if t[n]["tp"] + t[n]["fn"] > 0]
    return dict(macro_f1_present=sum(t[n]["f1"] for n in have) / len(have) if have else None, classes_present=have,
                images_per_class={n: t[n]["tp"] + t[n]["fn"] for n in NAMES})


def verdict(mean, selected, paper):
    if mean >= paper + MARGIN and selected >= paper + MARGIN:
        return "better"
    if mean <= paper - MARGIN and selected <= paper - MARGIN:
        return "worse"
    return "within_resolution"


def assert_finished(s):
    if not a.expect_epochs:
        return
    assert os.path.exists(f"runs/{a.run_prefix}{s}/train_time.json"), f"시드 {s} 학습이 끝나지 않았다"
    with open(f"runs/{a.run_prefix}{s}/results.csv") as fp:
        n = sum(1 for _ in fp) - 1
    assert n == a.expect_epochs, f"시드 {s}: {n} epoch"


def save_rows(rows, path):
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["stem", "gt", *[f"plain_{n}" for n in NAMES], *[f"tile_{n}" for n in NAMES]])
        for r in rows:
            w.writerow([r["stem"], ";".join(NAMES[c] for c in r["gt"]), *[round(v, 5) for v in r["plain"]], *[round(v, 5) for v in r["tile"]]])


def mean_std(xs):
    return dict(mean=statistics.mean(xs), std=statistics.stdev(xs) if len(xs) > 1 else 0.0, values=xs)


if a.stage == "val":
    assert not os.path.exists(STARTED) and not os.path.exists(DONE), "테스트를 시작한 뒤에는 검증 단계를 다시 돌리지 않는다"
    assert not os.path.exists(PROTO), "protocol.json이 이미 있다"
    res, search = {}, []
    for s in SEEDS:
        assert_finished(s)
        digest = sha(weights(s))
        model = YOLO(weights(s))
        rows = score_split(model, "val", True)
        save_rows(rows, f"{a.out}/val_conf_s{s}.csv")
        for mode in ("plain", "tile"):
            for c, n in enumerate(NAMES):
                search += [dict(seed=s, mode=mode, cls=n, **onoff(rows, mode, c, t)) for t in GRID]
        thr = {m: per_class_thresholds(rows, m, GRID) for m in ("plain", "tile")}
        lec = per_class_thresholds(rows, "plain", LECTURE_GRID)
        com = common_threshold(rows, "plain", GRID)
        res[s] = dict(weights_sha256=digest, thresholds=thr, common_threshold=com, lecture_grid_thresholds=lec,
                      macro_f1=dict(plain=table(rows, "plain", thr["plain"])["macro_f1"], tile=table(rows, "tile", thr["tile"])["macro_f1"],
                                    plain_lecture_grid=table(rows, "plain", lec)["macro_f1"],
                                    plain_common=table(rows, "plain", {n: com for n in NAMES})["macro_f1"]),
                      onoff_plain=table(rows, "plain", thr["plain"]), onoff_tile=table(rows, "tile", thr["tile"]),
                      **mask_map(model, "val", f"val_map_s{s}"))
    with open(f"{a.out}/val_threshold_search.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(search[0]))
        w.writeheader()
        w.writerows(search)
    mean_plain = statistics.mean(res[s]["macro_f1"]["plain"] for s in SEEDS)
    mean_tile = statistics.mean(res[s]["macro_f1"]["tile"] for s in SEEDS)
    mode = "tile" if mean_tile >= mean_plain + MARGIN else "plain"  # 해상도보다 작은 차이는 채택 근거가 아니다
    score = {s: (res[s]["mask_map50"] + res[s]["macro_f1"][mode]) / 2 for s in SEEDS}
    selected = max(SEEDS, key=lambda s: score[s])  # 동점이면 먼저 나온 시드
    with open(PROTO, "x") as proto_fp:
      json.dump(dict(fixed_at_utc=datetime.datetime.utcnow().isoformat(),
                   rule=dict(mode=f"시드 평균 검증 macro F1이 {MARGIN} 이상 높을 때만 타일 채택", seed="(검증 mask mAP50 + 검증 macro F1)/2 최대",
                             submission="대표 시드 모델을 제출 모델로 한다. 테스트 수치로 바꾸지 않으며 runs/paper와 runs/step3는 후보가 아니다",
                             threshold="클래스별 검증 F1 최대, 동점이면 낮은 값", grid=GRID, lecture_grid=LECTURE_GRID,
                             grid_note="논문 재현 실험의 검증 탐색에서 crack과 spall F1이 0.3까지 평평해 범위를 넓혔다. 그 실험의 테스트를 본 뒤 정한 값이므로 강의 범위 결과도 함께 기록한다",
                             claim=f"지표별로 테스트 시드 평균과 대표 시드가 모두 논문값+{MARGIN} 이상이면 better, 모두 논문값-{MARGIN} 이하면 worse, 그 외는 within_resolution"),
                   paper=PAPER, mean_val_macro_f1=dict(plain=mean_plain, tile=mean_tile), selected_mode=mode, selected_seed=selected, selection_score=score, seeds=res),
              proto_fp, indent=1, ensure_ascii=False)
    print(json.dumps(dict(selected_mode=mode, selected_seed=selected, score=score, mean_plain=mean_plain, mean_tile=mean_tile), ensure_ascii=False))
    sys.exit(0)

# 여기부터 test 단계: protocol.json을 읽기만 한다
proto = json.load(open(PROTO))
DONE = DONE.replace("test_results", "dry_results") if a.dry_on_val else DONE
assert not os.path.exists(DONE), "테스트 평가는 이미 끝났다. 다시 돌리지 않는다"
split = "val" if a.dry_on_val else "test"
mode = "tile" if a.dry_force_tile and a.dry_on_val else proto["selected_mode"]
proto_sha = sha(PROTO)
if not a.dry_on_val:
    # 시작을 먼저 기록한다. 중간에 멈춘 뒤 다시 돌리면 사유가 남는다
    if os.path.exists(STARTED):
        assert a.retry_reason, "이미 시작한 테스트다. 중단 후 재실행이면 --retry-reason을 준다"
    else:
        with open(STARTED, "x") as fp:
            json.dump(dict(started_utc=datetime.datetime.utcnow().isoformat(), protocol_sha256=proto_sha), fp)
    assert json.load(open(STARTED))["protocol_sha256"] == proto_sha, "시작 뒤 protocol.json이 바뀌었다"
    with open(f"{a.out}/test_attempts.log", "a") as fp:
        fp.write(f"{datetime.datetime.utcnow().isoformat()} {a.retry_reason or 'first run'}\n")
train_videos = {video_id(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f"{ROOT}/images/train/*.jpg")}
out = {}
for s in proto["seeds"]:
    p = proto["seeds"][s]
    assert sha(weights(s)) == p["weights_sha256"], "고정한 가중치와 다르다"
    model = YOLO(weights(s))
    rows = score_split(model, split, mode == "tile")
    save_rows(rows, f"{a.out}/{split if not a.dry_on_val else 'dry'}_conf_s{s}.csv")
    new = [r for r in rows if video_id(r["stem"]) not in train_videos]
    sub = f"{a.out}/{'dry' if a.dry_on_val else 'test'}_unseen_video.txt"
    with open(sub, "w") as fp:
        fp.write("".join(f"{ROOT}/images/{split}/{r['stem']}.jpg\n" for r in new))
    sub_yaml = f"{a.out}/{'dry' if a.dry_on_val else 'test'}_unseen_video.yaml"
    yaml.safe_dump(dict(path=ROOT, train="images/train", val="images/val", test=os.path.abspath(sub), names=cfg["names"]), open(sub_yaml, "w"))
    out[s] = dict(thresholds=p["thresholds"][mode], common_threshold=p["common_threshold"],
                  onoff=table(rows, mode, p["thresholds"][mode]),
                  macro_f1_plain=table(rows, "plain", p["thresholds"]["plain"])["macro_f1"],
                  macro_f1_common=table(rows, "plain", {n: p["common_threshold"] for n in NAMES})["macro_f1"],
                  unseen_video=dict(images=len(new), **present_macro_f1(new, mode, p["thresholds"][mode]),
                                    **(mask_map(model, "test", f"tmp_unseen_s{s}", sub_yaml) if new else {})),
                  **mask_map(model, split, f"{split}_map_s{s}"))
seeds = list(out)
summary = {k: mean_std(v) for k, v in dict(mask_map50=[out[s]["mask_map50"] for s in seeds], mask_map50_95=[out[s]["mask_map50_95"] for s in seeds],
                                            macro_f1=[out[s]["onoff"]["macro_f1"] for s in seeds], macro_f1_common=[out[s]["macro_f1_common"] for s in seeds],
                                            macro_f1_plain=[out[s]["macro_f1_plain"] for s in seeds]).items()}
sel = out[proto["selected_seed"]]
claim = {k: verdict(summary[k]["mean"], sel["onoff"]["macro_f1"] if k == "macro_f1" else sel[k], v) for k, v in PAPER.items()}
json.dump(dict(split=split, protocol_sha256=proto_sha, protocol_fixed_at_utc=proto["fixed_at_utc"], finished_utc=datetime.datetime.utcnow().isoformat(),
               selected_seed=proto["selected_seed"], selected_mode=mode, paper=PAPER, margin=MARGIN, verdict=claim, summary=summary, seeds=out),
          open(DONE, "w"), indent=1, ensure_ascii=False)
print(json.dumps(dict(summary=summary, claim=claim), ensure_ascii=False))
