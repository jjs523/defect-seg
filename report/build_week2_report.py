"""results/week2/ 산출물로 Step 3 2주차 보고서를 만든다. 실행: python report/build_week2_report.py --pdf
직접 주석 결과가 없으면 파일 이름과 본문에 미완료를 표시한 초안만 만든다."""
import argparse, base64, csv, datetime, hashlib, html as H, json, os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--pdf", action="store_true"); ap.add_argument("--annot-dir", default=os.path.join(ROOT, "results", "week2", "annotation")); ap.add_argument("--out-dir", default=os.path.join(ROOT, "report"))
a = ap.parse_args()
R = os.path.join(ROOT, "results", "week2"); WIN_TMP = "/mnt/c/Users/user/semg_report_tmp"; CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
CL = ["crack", "efflorescence", "spall_detach"]
ig = json.load(open(f"{R}/integrity.json")); chk = json.load(open(f"{R}/split_checks.json")); val = json.load(open(f"{R}/best_pt_val_check.json"))
csvr = list(csv.DictReader(open(f"{R}/train/results.csv"))); fmt = open(f"{R}/format_check.txt", encoding="utf-8").read()
sp = os.path.join(a.annot_dir, "annotation_summary.json"); ann = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
batch = list(csv.DictReader(open(os.path.join(ROOT, "annotation", "batch_list.csv"), encoding="utf-8")))
sha = hashlib.sha256(open(f"{R}/train/weights/best.pt", "rb").read()).hexdigest(); tt = json.load(open(f"{R}/train/train_time.json"))
g = lambda e, k: float(csvr[e - 1][k]); n_ep = len(csvr)
fit = lambda e: g(e, "metrics/mAP50-95(B)") + g(e, "metrics/mAP50-95(M)")
best = max(range(1, n_ep + 1), key=fit)
vmin = min(range(1, n_ep + 1), key=lambda e: g(e, "val/seg_loss")); m50 = max(range(1, n_ep + 1), key=lambda e: g(e, "metrics/mAP50(M)"))
rng = [g(e, "metrics/mAP50(M)") for e in range(2, n_ep + 1)]; cw = chk["crack_width"]

def b64(path):
    return f"data:image/{'jpeg' if path.endswith('jpg') else 'png'};base64," + base64.b64encode(open(path, "rb").read()).decode()
def tbl(head, body, cap=""):
    t = "<div class='tbl'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
    t += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>"
    return t + (f"<p class='cap'>{cap}</p>" if cap else "") + "</div>"
def excerpt(rel, starts, mark=()):
    """소스에서 해당 문자열로 시작하는 줄을 그대로 뽑는다. 내용이 어긋나면 바로 실패한다."""
    lines = open(os.path.join(ROOT, rel), encoding="utf-8").read().splitlines(); out = []
    for s in starts:
        hit = [l for l in lines if l.strip().startswith(s)]
        assert hit, f"{rel}: {s}"
        e = H.escape(hit[0]); out.append(f"<mark>{e}</mark>" if any(m in hit[0] for m in mark) else e)
    return "\n".join(out)

code_common = excerpt("src/common.py", ["def origin_id(stem):", 'm = re.match(', "return m.group(1) if m else stem"], mark=["re.match"])
code_split = excerpt("src/week2_prepare.py", ['held = {origin_id', 'for split in ["train", "val", "test"]:', 'for p in imgs:', 'reason = None', 'if p.stat().st_size == 0', 'reason = "unreadable"', 'elif split == "train" and origin_id', 'reason = "origin_in_val_or_test"', 'if reason:', 'seen[split].add(', 'report["origin_overlap_after"] =', 'assert not any('], mark=["held = {origin_id", "origin_id(p.stem) in held", "origin_overlap_after", "assert not any"])

tr, va, te = ig["train"], ig["val"], ig["test"]
inst_tbl = tbl(["항목", "train", "val", "test"], [
    ["제공 파일", f"{tr['files']:,}", va["files"], te["files"]], ["사용 영상", f"{tr['kept']:,}", va["kept"], te["kept"]], ["원본 프레임", tr["origins"], va["origins"], te["origins"]],
] + [[f"{c}: 영상 / 인스턴스", *[f"{d['images_per_class'][c]:,} / {d['instances_per_class'][c]:,}" for d in (tr, va, te)]] for c in CL], "표 3. 분할 후 데이터 구성 (results/week2/integrity.json)")
check_tbl = tbl(["점검 항목", "결과", "조치"], [
    ["0바이트 영상(읽기 실패)", f"train {tr['unreadable']}장 (URT020 원본 프레임 3개)", "<b>학습에서 제외</b>. 같은 원본의 다른 파일은 남아 있음"],
    ["val·test와 같은 원본의 train 영상", f"{tr['origin_in_val_or_test']}장 (<code class='sm'>URT017_2020_02_08_001_24512</code>)", "<b>train에서 제외</b>. val·test는 변경하지 않음"],
    ["영상 없는 라벨 / 라벨 없는 영상", "0 / 0", "조치 없음"], ["빈 라벨", "0", "조치 없음. 배경 전용 영상이 없다는 뜻이라 3주차 오류 분석에서 고려"],
    ["클래스 ID가 0·1·2 밖", "0", "조치 없음"], ["좌표가 0~1 밖 / 폴리곤 점 3쌍 미만", "0 / 0", "조치 없음"],
    ["data.yaml 경로", "제공본은 다른 PC 경로", "<b>실제 경로로 새로 작성</b>, 클래스 순서 0 crack, 1 efflorescence, 2 spall_detach 확인"],
], "표 2. 무결성 점검 결과와 조치. 강의 11쪽 코드도 정리된 dataset/ 전체에서 이상 파일 0")
loss_tbl = tbl(["epoch", "box", "seg", "cls", "dfl", "val seg", "val mask mAP50", "val mask mAP50-95"], [
    [f"{e}{' (best.pt)' if e == best else ''}", *[f"{g(e, k):.3f}" for k in ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss", "val/seg_loss", "metrics/mAP50(M)", "metrics/mAP50-95(M)"]]] for e in sorted({1, vmin, m50, best, n_ep})], "표 4. 주요 epoch의 손실과 검증 성능 (results/week2/train/results.csv)")
val_tbl = tbl(["", "전체", *CL], [["mask mAP50", f"{val['mask_map50']:.3f}", *[f"{val['per_class'][c]['map50']:.3f}" for c in CL]], ["mask mAP50-95", f"{val['mask_map']:.3f}", *[f"{val['per_class'][c]['map']:.3f}" for c in CL]]], "표 5. best.pt를 검증셋 40장에 다시 평가한 결과")

if ann:
    fl = ann["first_line"]
    ann_html = f"""<p>{H.escape(ann['tool'])}로 train 영상 <b>{ann['images']}장</b>에 직접 폴리곤을 그렸고, 인스턴스는 총 <b>{ann['instances_total']}개</b>다. 1주차 주석 기준서를 따랐고, 제공 라벨은 보지 않은 채 영상만 올려 작업했다. CVAT의 Segment Anything 3 보조 도구로 윤곽 후보를 만든 뒤, 남길 객체와 경계는 기준서에 따라 직접 판단하고 고쳤다. 영상은 train 원본 프레임에서 시드 42로 클래스별 5장을 촬영본이 겹치지 않게 고른 묶음 15장 안에서 주석했다(<code>annotation/batch_list.csv</code>). 판단 보류 목록의 영상은 묶음에서 뺐다.</p>
{tbl(["클래스", "영상 수", "인스턴스 수"], [[c, ann['images_per_class'].get(c, 0), ann['instances_per_class'].get(c, 0)] for c in CL] + [["합계", ann['images'], ann['instances_total']]], "표 1. 직접 주석 결과. 한 영상에 여러 클래스를 그리면 영상 수는 클래스마다 셈")}
<p>직접 주석을 제공 라벨과 비교한 IoU(영상별, 제공 라벨의 클래스 기준) 중앙값은 {", ".join(f"{k} {v['median']:.2f} ({v['n']}장, 범위 {v['min']:.2f}~{v['max']:.2f})" for k, v in ann['iou_with_provided'].items())}다. 이 값은 정답률이 아니라 두 주석의 일치 정도다. 영상별 값은 <code>results/week2/annotation/annotation_compare.csv</code>에 있다. 직접 주석은 라벨링 절차 연습이 목적이라 학습에는 넣지 않았고, 학습에는 제공 라벨을 사용했다.</p>
<figure><img src="{b64(os.path.join(a.annot_dir, 'annotation_overlay.jpg'))}" style="width:100%"><figcaption>그림 1. 직접 주석(빨강 crack, 파랑 efflorescence, 초록 spall_detach)과 제공 라벨(흰 선)</figcaption></figure>"""
    rl = ann.get("relabeled") or []
    if rl:
        n_rl = sum(r["polygons"] for r in rl)
        ann_html += f"""<p>내보낸 라벨을 영상에 겹쳐 점검하다가 영상 {len(rl)}장의 균열 폴리곤 {n_rl}개가 AI 도구 패널의 기본 라벨(spall_detach)로 들어간 것을 발견했다. 좌표는 그대로 두고 클래스 번호만 crack으로 고쳤다(<code>annotation/label_fixes.csv</code>, <code>src/week2_annotation.py --relabel</code>). 표 1과 IoU는 고친 뒤 값이고, CVAT에서 내보낸 원본은 <code>results/week2/annotation/export_original.zip</code>에 그대로 있다.</p>"""
        ann_chk = f"<p>직접 주석을 내보낸 파일도 같은 항목으로 점검했다. 형식 오류는 0개였고, 클래스 오지정 {n_rl}개(영상 {len(rl)}장)는 위 1절처럼 클래스 번호만 고쳤다.</p>"
    else:
        ann_chk = "<p>직접 주석을 내보낸 파일도 같은 항목으로 점검했고 이상 항목은 0개였다.</p>"
    ann_fmt = f"<p>직접 주석을 내보낸 파일도 같은 방식으로 확인했고 Segmentation 포맷이다.</p><pre>{H.escape(fl['file'])}\n토큰 개수: {fl['tokens']}\n앞부분: {H.escape(str(fl['head']))}</pre>"
else:
    ann_html = f"""<div class="todo"><b>[미완료] 직접 주석 결과가 아직 없다.</b> 묶음 {len(batch)}장을 주석하고 내보낸 파일을 넣어 다시 빌드하면 이 자리가 표와 그림으로 바뀐다.</div>"""
    ann_fmt = ""; ann_chk = ""

today = datetime.date.today()
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 2주차 보고서</title><style>
@page {{ size: A4; margin: 20mm 18mm 18mm 18mm; }}
body {{ font-family: "Noto Serif KR", "Batang", "바탕", serif; font-size: 10.2pt; line-height: 1.6; color: #000; margin: 0; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 8pt; }} h2 {{ font-size: 12.5pt; margin: 15pt 0 5pt; border-bottom: 1px solid #000; padding-bottom: 2pt; break-after: avoid; }}
p {{ margin: 3pt 0 5pt; }} table {{ border-collapse: collapse; width: 100%; font-size: 8.9pt; }} th, td {{ border: 1px solid #555; padding: 2.5pt 4pt; vertical-align: middle; word-break: keep-all; }} th {{ background: #eee; }}
.tbl {{ break-inside: avoid; margin: 6pt 0; }} tr {{ break-inside: avoid; }} .cap {{ font-size: 8.8pt; text-align: center; margin: 3pt 0 6pt; }}
figure {{ margin: 6pt 0; text-align: center; break-inside: avoid; }} figure img {{ max-width: 100%; }} figcaption {{ font-size: 8.8pt; margin-top: 2pt; }}
code {{ font-family: "D2Coding", "Consolas", monospace; font-size: 8.8pt; }} code.sm {{ font-size: 7.4pt; word-break: break-all; }} pre {{ font-family: "D2Coding", "Consolas", monospace; font-size: 8.3pt; line-height: 1.35; border: 1px solid #999; padding: 5pt; white-space: pre-wrap; margin: 4pt 0; break-inside: avoid; }}
mark {{ background: #ffe58a; }} .hdr th {{ width: 14%; }} ul {{ margin: 2pt 0 5pt; padding-left: 15pt; }} li {{ margin: 1pt 0; }}
.todo {{ border: 2px solid #c00; color: #c00; padding: 8pt; margin: 6pt 0; }}
</style></head><body>
<h1>Step 3 2주차 과제 보고서</h1>
<table class="hdr"><tr><th>학 과</th><td>컴퓨터소프트웨어학부</td><th>학 번</th><td>22114553</td></tr><tr><th>이 름</th><td>전진성</td><th>제출일</th><td>{today.year}년 {today.month}월 {today.day}일</td></tr></table>

<h2>요약</h2>
<p>제공 데이터를 점검해 읽을 수 없는 영상 {tr['unreadable']}장과 test와 원본이 같은 train 영상 {tr['origin_in_val_or_test']}장을 뺀 뒤, train {tr['kept']:,}장으로 YOLOv8s-Seg를 학습했다. 학습은 강의 설정 그대로 오류 없이 돌았고, 조기 종료로 60 epoch 중 {n_ep} epoch에서 멈췄다. best.pt(epoch {best})의 검증셋 mask mAP50은 {val['mask_map50']:.3f}, mAP50-95는 {val['mask_map']:.3f}다. 다만 검증 손실이 epoch {vmin} 뒤로 더 낮아지지 않아 과적합 징후가 있고, 검증셋이 40장이라 이 값은 흔들림이 크다. 이 보고서는 학습 파이프라인 확인이 범위이며, 이 가중치를 test에 평가하지 않았다.</p>

<h2>1. 주석 결과 (제출물 ①)</h2>
{ann_html}

<h2>2. 포맷 확인 (제출물 ②)</h2>
<p>강의 10쪽 코드를 정리된 데이터셋 전체에 실행한 출력이다. 클래스 ID 뒤에 좌표 쌍이 이어지므로 Segmentation 포맷이다.</p>
<pre>{H.escape(fmt)}</pre>
{ann_fmt}

<h2>3. 무결성 점검 (제출물 ③)</h2>
<p>영상과 라벨의 대응, 빈 파일, 클래스 ID, 좌표 범위, 폴리곤 점 수를 점검했다. 이상 항목은 학습 전에 고쳤다.</p>
{check_tbl}
{ann_chk}
<pre>{H.escape(open(os.path.join(ROOT, "dataset", "data.yaml"), encoding="utf-8").read())}</pre>
<p class="cap">dataset/data.yaml</p>
{inst_tbl}

<h2>4. 분할 코드 (제출물 ④)</h2>
<p>이 과제는 train 4,036 / val 40 / test 40으로 나뉜 데이터가 제공되어 강의의 8:1:1 무작위 분할 대신 제공 분할을 유지했다. 그래서 val·test는 그대로 두고, <b>같은 원본에서 나온 영상을 train에서 빼는 방식</b>으로 원본 단위 분리를 맞췄다. 표시한 줄이 원본 ID 기준으로 처리하는 부분이다.</p>
<pre>{code_common}</pre>
<pre>{code_split}</pre>
<p class="cap">코드 1. src/common.py와 src/week2_prepare.py에서 원본 단위 분리에 해당하는 줄만 뽑은 것(중간 줄 생략)</p>
<p>강의의 <code>stem.split("_aug")[0]</code>는 이 데이터에 맞지 않아 파일명에서 원본 프레임 번호까지 정규식으로 뽑도록 고쳤다. val 24장과 test 19장에는 Roboflow 내보내기 꼬리(<code>_jpg.rf.…</code>)가 붙어 있고 train과 겹치는 test 파일도 그중 하나라, 강의 함수로는 겹치는 train 파일을 {chk['train_overlap_files_lecture_id']}개 찾지만 고친 함수로는 {chk['train_overlap_files_our_id']}개를 찾는다. 정규식에 맞지 않는 파일명은 전체 분할에서 0개다.</p>
<ul><li><b>분리 확인</b>: 제외 후 train과 val, train과 test가 공유하는 원본 프레임은 모두 0개다. 코드에 assert를 넣어 겹치면 멈추게 했다.</li>
<li><b>증강은 train에만</b>: 증강 이름(<code>_aug</code>, <code>flip</code>, <code>rot</code>)이 붙은 파일은 val·test에 0개다.</li>
<li><b>남은 한계</b>: 프레임은 겹치지 않아도 같은 촬영본은 공유한다. train과 촬영본이 겹치는 경우가 val {chk['shared_video']['val']['shared']}/{chk['shared_video']['val']['total']}개, test {chk['shared_video']['test']['shared']}/{chk['shared_video']['test']['total']}개다. 제공 분할을 유지했으므로 이 부분은 그대로이며, 검증·테스트 성능이 새 촬영본에서의 성능보다 낙관적일 수 있다.</li></ul>

<h2>5. 학습 곡선 (제출물 ⑤)</h2>
<figure><img src="{b64(f'{R}/curve.png')}" style="width:100%"><figcaption>그림 2. curve.png: 분할 손실과 검증셋 mask mAP (점선: best.pt)</figcaption></figure>
{loss_tbl}
<ul><li><b>정상 동작</b>: 첫 epoch부터 box·seg·cls·dfl 손실이 모두 출력되었고, train seg 손실은 {g(1, 'train/seg_loss'):.2f}에서 {g(n_ep, 'train/seg_loss'):.2f}로 꾸준히 줄었다. mask mAP도 오르므로 라벨 포맷 문제는 없다고 본다.</li>
<li><b>과적합 징후</b>: val seg 손실은 epoch {vmin}의 {g(vmin, 'val/seg_loss'):.2f}가 최저이고, 그 뒤로는 이보다 낮아지지 않으며 epoch {n_ep}에는 {g(n_ep, 'val/seg_loss'):.2f}다. train 손실만 계속 줄어드는 상태다.</li>
<li><b>검증 지표의 흔들림</b>: mask mAP50은 epoch 2 이후 {min(rng):.2f}~{max(rng):.2f} 사이를 오간다. 검증셋이 40장이라 epoch 간 작은 차이는 믿기 어렵다.</li>
<li><b>모델 선택</b>: Ultralytics 8.4.198은 box mAP50-95와 mask mAP50-95의 합이 가장 큰 epoch를 best.pt로 저장한다. 이 값은 epoch {best}가 {fit(best):.3f}로 가장 크고 epoch {m50}는 {fit(m50):.3f}다. mAP50은 이 기준에 들어가지 않아, mask mAP50이 가장 높은 epoch {m50}({g(m50, 'metrics/mAP50(M)'):.3f})는 선택되지 않았다.</li>
<li><b>학습 시간</b>: {tt['seconds']:,}초(약 {tt['seconds'] / 60:.0f}분). RTX 4090 서버, patience 20으로 epoch {best} 뒤 20 epoch 동안 개선이 없어 epoch {n_ep}에서 멈췄다.</li></ul>
{val_tbl}
<p>best.pt가 정상으로 저장되었는지 보려고 RTX 2070 SUPER PC에서 다시 평가했더니 results.csv의 epoch {best} 값(mask mAP50 {g(best, 'metrics/mAP50(M)'):.3f}, mAP50-95 {g(best, 'metrics/mAP50-95(M)'):.3f})과 소수 셋째 자리까지 같았다. efflorescence가 가장 낮지만, 클래스별 인스턴스가 10~27개뿐이라 원인은 3주차 오류 분석에서 확인한다.</p>
<p>이 best.pt와 별도로, 논문 조건 재현 실험 1회와 조기 종료 없이 60 epoch까지 학습한 추가 실험(시드 42, 1, 2)을 진행했다. 이 실험들은 검증셋으로 임계값과 대표 시드를 고정한 뒤 test를 평가했고, 결과는 3주차 보고서에서 다룬다. 이 2주차 best.pt는 그 후보가 아니며 test에 평가하지 않았다.</p>

<h2>6. 설정 근거 (제출물 ⑥)</h2>
<ul><li><b>imgsz 960</b>: 영상 긴 변이 1920이라 960이면 1/2, 기본값 640이면 1/3로 줄어든다. 가는 균열의 세부를 더 남기려고 강의·논문 조건인 960을 썼고, 메모리 사용량을 고려해 batch는 4로 했다. 다만 제공 crack 폴리곤의 평균 폭(2×면적/둘레, 원본 {cw['frames']}프레임의 인스턴스 {cw['instances']}개)은 중앙값이 1920에서 {cw['px_at_1920']['q50']} px, 640에서도 {cw['px_at_640']['q50']} px다. 라벨 띠가 실제 균열선보다 넓어서 이 값이 균열선 폭은 아니며, 이 수치만으로 960이 꼭 필요하다고 말할 수는 없다.</li>
<li><b>Mosaic 비활성화</b>: 여러 영상을 이어 붙이면 이음부·배관과 결함의 공간적 맥락이 끊겨, 결함과 비슷한 선형 구조를 구분하는 데 불리할 수 있다.</li>
<li><b>MixUp 비활성화</b>: 두 영상을 겹치면 결함 경계가 흐려지고, 희미한 얼룩처럼 보이는 합성 영상이 생겨 경계 학습과 해석에 불리할 수 있다.</li>
<li><b>유지한 증강</b>: 그 밖에는 Ultralytics 기본값(hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, translate 0.1, scale 0.5, fliplr 0.5)을 썼다. train의 대부분이 이미 오프라인 증강본이라는 점도 함께 고려해야 한다.</li>
<li><b>한계</b>: 960과 640, Mosaic·MixUp 사용과 비사용을 이 과제에서 직접 비교하지는 않았다. 위 내용은 효과를 측정한 결과가 아니라 강의·논문 조건을 따른 설계 근거다.</li></ul>

<h2>7. 재현 정보와 제출물 대조</h2>
<pre># 제출 zip을 푼 뒤 code_results/ 안에서 실행. 제공 데이터는 data/에 푼다
bash setup.sh &amp;&amp; source .venv/bin/activate
mkdir -p results/week2/train/weights &amp;&amp; cp ../best.pt results/week2/train/weights/
# 제출한 best.pt로 데이터 정리·점검·곡선·보고서·제출 zip을 다시 만든다(학습 없음)
python main.py week2
# 학습부터 다시 할 때: 데이터 정리 → 학습 → 결과 복사 → 위 명령
python src/week2_prepare.py
python src/week2_train.py --epochs 60 --batch 4 --name step3
cp -r runs/step3/. results/week2/train/
yolov8s-seg.pt, imgsz 960, batch 4, AdamW, lr0 0.00015, mosaic 0, mixup 0, patience 20, seed 42
best.pt sha256 {sha}</pre>
<p>학습은 RTX 4090 서버(Python 3.11.18, torch 2.14.1, ultralytics 8.4.198), 재평가는 RTX 2070 SUPER PC(Python 3.10.12, torch 2.14.1, ultralytics 8.4.198)에서 했다. 두 곳의 dataset/은 영상 수·라벨 내용·영상 파일 이름의 해시가 일치하고, best.pt 해시도 서버 원본과 일치한다. 패키지 버전은 <code>requirements.lock.txt</code>에 고정했다. <code>main.py week2</code>는 재평가에 GPU, 곡선에 나눔고딕, PDF 생성에 WSL의 Windows Chrome을 사용한다. 직접 주석 원본은 <code>results/week2/annotation/export_original.zip</code>에 있다.</p>
{tbl(["강의 제출물", "산출물"], [["① 주석 결과", "1절, results/week2/annotation/" + ("" if ann else " <b>(미완료)</b>")], ["② 포맷 확인", "2절, results/week2/format_check.txt"], ["③ 무결성 점검", "3절 표 2·3, results/week2/integrity.json · excluded_images.csv"], ["④ 분할 코드", "4절 코드 1, src/common.py · src/week2_prepare.py"], ["⑤ 학습 곡선", "5절 그림 2, results/week2/curve.png · train/results.png"], ["⑥ 설정 근거", "6절"], ["가중치", "best.pt"]])}
</body></html>"""
os.makedirs(a.out_dir, exist_ok=True)
hp = os.path.join(a.out_dir, "week2_report.html"); open(hp, "w", encoding="utf-8").write(html); print("HTML", len(html) // 1024, "KB")
if a.pdf:
    os.makedirs(WIN_TMP, exist_ok=True); shutil.copy(hp, f"{WIN_TMP}/week2_report.html")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", r"--print-to-pdf=C:\Users\user\semg_report_tmp\week2_report.pdf", "--virtual-time-budget=10000", "file:///C:/Users/user/semg_report_tmp/week2_report.html"], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    name = "Step3_2주차과제_보고서_22114553_전진성" + ("" if ann else "_초안_주석미완료") + ".pdf"
    out = os.path.join(a.out_dir, name); shutil.copy(f"{WIN_TMP}/week2_report.pdf", out); shutil.rmtree(WIN_TMP); print("PDF", out, os.path.getsize(out) // 1024, "KB")
