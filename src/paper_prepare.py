"""논문 조건 재현용 dataset_paper/를 만든다. 제공 분할(4,036/40/40)을 그대로 두고 읽을 수 없는 0바이트 영상만 뺀다."""
import collections, json, os
from pathlib import Path

SRC, DST = Path("data"), Path("dataset_paper")
report = {}
for split in ["train", "val", "test"]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    r = collections.Counter()
    images_per_class = collections.Counter()
    for p in sorted((SRC / "images" / split).glob("*.jpg")):
        r["files"] += 1
        if p.stat().st_size == 0:
            r["zero_byte"] += 1
            continue
        lab = SRC / "labels" / split / f"{p.stem}.txt"
        ids = [int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()]
        r["kept"] += 1
        r["instances"] += len(ids)
        images_per_class.update(set(ids))
        for s, d in [(p, DST / "images" / split / p.name), (lab, DST / "labels" / split / lab.name)]:
            if not d.exists():
                os.link(s, d)
    report[split] = dict(r, images_per_class={str(k): v for k, v in sorted(images_per_class.items())})
(DST / "data.yaml").write_text(f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crack\n  1: efflorescence\n  2: spall_detach\n")
os.makedirs("results/paper", exist_ok=True)
json.dump(report, open("results/paper/dataset.json", "w"), indent=1)
print(json.dumps(report))
