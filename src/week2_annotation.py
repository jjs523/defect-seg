"""2주차 직접 주석 지원.
pick   : train 원본에서 주석할 영상을 고르고 CVAT·Roboflow에 올릴 zip을 만든다(라벨은 넣지 않는다).
import : 내보낸 YOLO seg 결과를 점검·집계하고 제공 라벨과 비교한다.
예: python src/week2_annotation.py pick
    python src/week2_annotation.py import --export 내보낸파일.zip --tool CVAT"""
import argparse, collections, csv, json, os, random, re, shutil, sys, zipfile
from pathlib import Path
import cv2, numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NAMES, SEED, origin_id

ANN = Path("annotation"); LIST = ANN / "batch_list.csv"; ZIP = ANN / "week2_annotation_images.zip"
TRAIN_I, TRAIN_L = Path("dataset/images/train"), Path("dataset/labels/train")
MIN_IMAGES, COLORS = 10, {0: (40, 40, 230), 1: (230, 120, 30), 2: (40, 170, 40)}


def read_label(text):
    return [(int(v[0]), np.array(v[1:], dtype=float).reshape(-1, 2)) for v in (l.split() for l in text.splitlines()) if v]


def pick(per_class):
    hold = {r["stem"] for r in csv.DictReader(open("results/week1/hold_list.csv", encoding="utf-8"))}
    by_class = collections.defaultdict(list)
    for p in sorted(TRAIN_I.glob("*.jpg")):
        if origin_id(p.stem) != p.stem or p.stem in hold:
            continue
        cls = {c for c, _ in read_label((TRAIN_L / f"{p.stem}.txt").read_text())}
        if len(cls) == 1:
            by_class[cls.pop()].append(p)
    rng, rows, videos = random.Random(SEED), [], set()
    for c in range(3):                      # 촬영본이 겹치지 않게 클래스별로 고른다
        pool = by_class[c][:]
        rng.shuffle(pool)
        got = 0
        for p in pool:
            video = p.stem.rsplit("_", 1)[0]
            if got < per_class and video not in videos:
                videos.add(video); got += 1
                rows.append(dict(stem=p.stem, provided_class=NAMES[c], video=video))
        assert got == per_class, f"{NAMES[c]} 후보 부족: {got}"
    ANN.mkdir(exist_ok=True)
    with open(LIST, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["stem", "provided_class", "video"]); w.writeheader(); w.writerows(rows)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            z.write(TRAIN_I / f"{r['stem']}.jpg", f"{r['stem']}.jpg")
    print(len(rows), "장", ZIP, os.path.getsize(ZIP) // 1024, "KB")


def load_export(path):
    """zip 또는 폴더에서 data.yaml과 txt 내용만 텍스트로 읽는다."""
    files = {}
    if os.path.isdir(path):
        for p in Path(path).rglob("*"):
            if p.suffix.lower() in (".txt", ".yaml", ".yml"):
                files[str(p)] = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith((".txt", ".yaml", ".yml")):
                    files[n.replace("\\", "/")] = z.read(n).decode("utf-8-sig", errors="replace")
    return files


def import_export(path, out, tool, relabel=None):
    batch = {r["stem"]: r for r in csv.DictReader(open(LIST, encoding="utf-8"))}
    files = load_export(path)
    yamls = [t for n, t in files.items() if n.lower().endswith((".yaml", ".yml"))]
    if not yamls:
        sys.exit("내보낸 파일에 data.yaml이 없어 클래스 순서를 확인할 수 없습니다.")
    meta = yaml.safe_load(yamls[0])
    names = meta.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if names != NAMES:
        sys.exit(f"클래스 순서가 다릅니다: {names}. crack, efflorescence, spall_detach 순서로 다시 등록하세요.")
    is_roboflow = "roboflow" in meta or any("_jpg.rf." in n for n in files)
    if is_roboflow != (tool == "Roboflow"):
        sys.exit(f"--tool {tool}이 내보낸 파일과 맞지 않습니다.")
    if any(Path(n).name.lower().startswith("readme") and re.search(r"\bFit\b|Augmentation", t) for n, t in files.items()):
        sys.exit("Roboflow 버전에 Fit 크기 조정이나 증강이 켜져 있습니다. 전처리 없이 다시 내보내세요.")
    found, ignored = collections.defaultdict(list), []
    for n in files:
        if not n.lower().endswith(".txt"):
            continue
        stem = re.sub(r"_jpg\.rf\..*$", "", Path(n).stem)    # Roboflow가 붙이는 꼬리 제거
        if stem in batch:
            found[stem].append(n)
        elif "labels" in Path(n).parts:
            ignored.append(Path(n).name)
    dup = sorted(s for s, v in found.items() if len(v) > 1)
    if dup:
        sys.exit(f"같은 영상의 라벨이 여러 개입니다({len(dup)}장, 예: {dup[0]}). 증강 없이 다시 내보내세요.")
    labels = {s: files[v[0]] for s, v in found.items()}
    relabeled = []
    if relabel:                                          # 내보낸 뒤 점검에서 찾은 클래스 오지정을 고친다(좌표는 그대로)
        for r in csv.DictReader(open(relabel, encoding="utf-8")):
            if r["stem"] not in labels:
                sys.exit(f"{relabel}: {r['stem']}의 라벨이 내보낸 파일에 없습니다.")
            src, dst = str(NAMES.index(r["from"])), str(NAMES.index(r["to"]))
            lines = labels[r["stem"]].splitlines()
            n = sum(1 for l in lines if l.split()[:1] == [src])
            labels[r["stem"]] = "\n".join(" ".join([dst] + l.split()[1:]) if l.split()[:1] == [src] else l for l in lines) + "\n"
            relabeled.append(dict(stem=r["stem"], **{"from": r["from"], "to": r["to"]}, polygons=n, reason=r["reason"]))
    if ignored:
        print(f"[주의] 주석 묶음에 없는 라벨 {len(ignored)}개는 집계하지 않습니다: {ignored[:5]}")
    if not labels:
        sys.exit("주석 묶음과 이름이 맞는 라벨 파일이 없습니다.")
    bad, first_stem = collections.Counter(), None
    parsed, inst, imgs, rows = {}, collections.Counter(), collections.Counter(), []
    for stem in sorted(labels):
        for l in labels[stem].splitlines():
            v = l.split()
            if not v:
                continue
            if len(v) == 5:
                sys.exit(f"{stem}: Detection 포맷입니다. Segmentation 포맷으로 다시 내보내세요.")
            if len(v) < 7 or (len(v) - 1) % 2:
                bad["bad_polygon"] += 1
            if v[0] not in ("0", "1", "2"):
                bad["bad_class_id"] += 1
            try:
                xy = np.array(v[1:], dtype=float)
            except ValueError:
                bad["bad_coord"] += 1
                continue
            if not ((0 <= xy) & (xy <= 1)).all():
                bad["bad_coord"] += 1
            elif len(xy) >= 6 and len(xy) % 2 == 0 and cv2.contourArea((xy.reshape(-1, 2) * [1920, 1080]).astype(np.float32)) == 0:
                bad["zero_area_polygon"] += 1
        if not bad:
            parsed[stem] = read_label(labels[stem])
    if bad:
        sys.exit(f"라벨 오류: {dict(bad)}")
    annotated = [s for s in parsed if parsed[s]]
    polys = [xy for s in annotated for _, xy in parsed[s]]
    if polys and all(len(xy) == 4 and len(np.unique(xy[:, 0])) == 2 and len(np.unique(xy[:, 1])) == 2 for xy in polys):
        sys.exit("모든 라벨이 사각형입니다. 폴리곤으로 주석했는지 확인하세요.")
    if len(annotated) < MIN_IMAGES:
        sys.exit(f"주석된 영상이 {len(annotated)}장입니다. 강의 목표는 10~20장입니다.")
    for stem in annotated:
        img = cv2.imread(str(TRAIN_I / f"{stem}.jpg"))
        H, W = img.shape[:2]
        prov = read_label((TRAIN_L / f"{stem}.txt").read_text())
        mine = parsed[stem]
        for c in {c for c, _ in mine}:
            imgs[NAMES[c]] += 1
        for c, _ in mine:
            inst[NAMES[c]] += 1
        pc = prov[0][0]
        def mask(items):
            m = np.zeros((H, W), np.uint8)
            for c, xy in items:
                if c == pc:
                    cv2.fillPoly(m, [np.round(xy * [W, H]).astype(np.int32)], 1)
            return m
        a, b = mask(mine) > 0, mask(prov) > 0
        rows.append(dict(stem=stem, provided_class=NAMES[pc], provided_instances=len(prov), my_instances=len(mine),
                         my_classes=",".join(sorted({NAMES[c] for c, _ in mine})),
                         iou_with_provided=round(float((a & b).sum() / max(1, (a | b).sum())), 3)))
    out = Path(out)
    shutil.rmtree(out / "labels", ignore_errors=True)    # 다시 가져올 때 이전 라벨이 섞이지 않게 한다
    (out / "labels").mkdir(parents=True)
    keep = out / "export_original.zip"                   # 재현용으로 내보낸 원본을 함께 보관
    if os.path.isdir(path):
        shutil.make_archive(str(keep.with_suffix("")), "zip", path)
    elif os.path.abspath(path) != os.path.abspath(keep):
        shutil.copy(path, keep)
    for stem in annotated:
        (out / "labels" / f"{stem}.txt").write_text(labels[stem])
    with open(out / "annotation_compare.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    first_stem = annotated[0]
    tok = next(l for l in labels[first_stem].splitlines() if l.strip()).split()
    iou = collections.defaultdict(list)
    for r in rows:
        iou[r["provided_class"]].append(r["iou_with_provided"])
    summary = dict(images=len(annotated), empty_files=sorted(set(parsed) - set(annotated)), images_per_class=dict(imgs),
                   instances_per_class=dict(inst), instances_total=sum(inst.values()),
                   iou_with_provided={k: dict(n=len(v), median=float(np.median(v)), min=min(v), max=max(v)) for k, v in sorted(iou.items())},
                   first_line=dict(file=f"{first_stem}.txt", tokens=len(tok), head=tok[:9]), source=os.path.basename(path), tool=tool, ignored_files=ignored, relabeled=relabeled)
    json.dump(summary, open(out / "annotation_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # 제공 라벨과 겹쳐 그린 그림: 흰 선 = 제공 라벨, 색 선 = 직접 주석
    tiles = []
    for stem in sorted(annotated, key=lambda s: (batch[s]["provided_class"], s))[::max(1, len(annotated) // 6)][:6]:
        img = cv2.imread(str(TRAIN_I / f"{stem}.jpg"))
        H, W = img.shape[:2]
        for c, xy in read_label((TRAIN_L / f"{stem}.txt").read_text()):
            cv2.polylines(img, [np.round(xy * [W, H]).astype(np.int32)], True, (255, 255, 255), 5, cv2.LINE_AA)
        for c, xy in parsed[stem]:
            cv2.polylines(img, [np.round(xy * [W, H]).astype(np.int32)], True, COLORS[c], 3, cv2.LINE_AA)
        s = 640 / max(H, W)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tile = np.full((360, 640, 3), 255, np.uint8)
        tile[(360 - img.shape[0]) // 2:(360 - img.shape[0]) // 2 + img.shape[0], (640 - img.shape[1]) // 2:(640 - img.shape[1]) // 2 + img.shape[1]] = img
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.full((360, 640, 3), 255, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(out / "annotation_overlay.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "import"])
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--export")
    ap.add_argument("--out", default="results/week2/annotation")
    ap.add_argument("--tool", choices=["CVAT", "Roboflow"])
    ap.add_argument("--relabel", help="클래스 오지정 수정 목록 csv(stem,from,to,reason). 내보낸 원본 zip은 그대로 보관")
    a = ap.parse_args()
    if a.cmd == "pick":
        pick(a.per_class)
    else:
        if not a.export or not a.tool:
            ap.error("--export와 --tool이 필요합니다")
        import_export(a.export, a.out, a.tool, a.relabel)
