"""Step 3 최종 리포트(3주차)와 주석 기준서 수정본을 만든다. 실행: python report/build_week3_report.py --pdf
입력은 results/improve(11/5 고정·테스트 결과), results/week3(week3_eval.py 출력), results/week2(데이터 점검·직접 주석)다."""
import argparse, base64, csv, datetime, html as H, json, os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--pdf", action="store_true"); a = ap.parse_args()
J = lambda p: json.load(open(os.path.join(ROOT, p), encoding="utf-8"))
W3, IMP = "results/week3", "results/improve"
s = J(f"{W3}/week3_summary.json"); done = J(f"{IMP}/test_results.json"); proto = J(f"{IMP}/protocol.json")
ig = J("results/week2/integrity.json"); ann = J("results/week2/annotation/annotation_summary.json")
cw = J("results/week2/split_checks.json")["crack_width"]; paper_run = J("results/paper/test_metrics.json")
dec = J(f"{W3}/model_decision.json")
CL = ["crack", "efflorescence", "spall_detach"]; THR = s["thresholds"]; T = s["test_onoff"]; V = s["val_onoff"]
rr = s["test_rerun"]; sm = done["summary"]; uv = done["seeds"]["1"]["unseen_video"]
WIN_TMP = "/mnt/c/Users/user/semg_report_tmp"; CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
curve = list(csv.DictReader(open(os.path.join(ROOT, f"{W3}/val_threshold_f1.csv"), encoding="utf-8")))
tr_rows = list(csv.DictReader(open(os.path.join(ROOT, f"{IMP}/train_s1/results.csv"))))
fit = lambda r: float(r["metrics/mAP50-95(B)"]) + float(r["metrics/mAP50-95(M)"])
best_ep = int(float(max(tr_rows, key=fit)["epoch"]))


def b64(p):
    p = os.path.join(ROOT, p)
    return f"data:image/{'jpeg' if p.endswith('jpg') else 'png'};base64," + base64.b64encode(open(p, "rb").read()).decode()


def tbl(head, body, cap=""):
    t = "<div class='tbl'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
    t += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>"
    return t + (f"<p class='cap'>{cap}</p>" if cap else "") + "</div>"


def excerpt(rel, starts, mark=()):
    lines = open(os.path.join(ROOT, rel), encoding="utf-8").read().splitlines(); out = []
    for st in starts:
        hit = [l for l in lines if l.strip().startswith(st)]
        assert hit, f"{rel}: {st}"
        e = H.escape(hit[0]); out.append(f"<mark>{e}</mark>" if any(m in hit[0] for m in mark) else e)
    return "\n".join(out)


f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.3f}"
code_thr = excerpt("src/improve_eval.py", [
    'rows = score_split(model, "val", True)', 'thr = {m: per_class_thresholds(rows, m, GRID)', 'out[n] = grid[scores.index(max(scores))]',
    'with open(PROTO, "x") as proto_fp:', '# 여기부터 test 단계', 'proto = json.load(open(PROTO))', 'assert not os.path.exists(DONE)',
    'assert sha(weights(s)) == p["weights_sha256"]', 'onoff=table(rows, mode, p["thresholds"][mode]),'],
    mark=['score_split(model, "val"', 'with open(PROTO, "x")', 'p["thresholds"][mode]'])
code_onoff = excerpt("src/improve_eval.py", ['r = model.predict(crop, imgsz=TILE, conf=CONF_FLOOR', 'best[c] = max(best[c], s)', 'g, q = cid in r["gt"], r[mode][cid] >= thr'],
                     mark=["conf=CONF_FLOOR", ">= thr"])

# --- 표 ---
data_tbl = tbl(["항목", "train", "val", "test"], [
    ["제공 파일", f"{ig['train']['files']:,}", ig['val']['files'], ig['test']['files']],
    ["사용 영상", f"{ig['train']['kept']:,}", ig['val']['kept'], ig['test']['kept']],
    ["원본 프레임", ig['train']['origins'], ig['val']['origins'], ig['test']['origins']]] +
    [[f"{c}: 영상 / 인스턴스", *[f"{ig[sp]['images_per_class'][c]:,} / {ig[sp]['instances_per_class'][c]:,}" for sp in ("train", "val", "test")]] for c in CL],
    "표 1. 데이터 구성. train은 0바이트 63장과 test와 원본이 같은 55장을 뺀 수")
set_tbl = tbl(["항목", "값", "근거"], [
    ["모델", "yolov8s-seg.pt (COCO 사전학습)", "논문 조건"], ["입력 크기", "imgsz 960", f"균열 폭 중앙값이 1920 기준 {cw['px_at_1920']['q50']} px → 960에서 {cw['px_at_960']['q50']} px, 640에서 {cw['px_at_640']['q50']} px(하위 10 % {cw['px_at_640']['q10']} px)"],
    ["학습", "60 epoch, patience 0, batch 4, AdamW lr0 0.00015", "논문 조건. 조기 종료 없이 끝까지 학습"],
    ["증강", "mosaic 0, mixup 0, 그 밖은 Ultralytics 기본", "모자이크는 가는 균열을 자르거나 줄이고, MixUp은 흰 석출물과 얼룩의 질감을 섞는다. train의 97 %가 이미 오프라인 증강본"],
    ["시드", "42, 1, 2 (deterministic)", "시드 차이를 보고 대표 시드를 검증셋으로 고름"],
    ["가중치 선택", f"검증 box+mask mAP50-95 합이 최대인 epoch {best_ep}", "Ultralytics best.pt 기준"]],
    "표 2. 제출 모델(clean_s1) 학습 설정")
sel_tbl = tbl(["시드", "검증 mask mAP50", "검증 macro F1", "선택 점수(평균)"], [
    [k + (" <b>(선택)</b>" if k == "1" else ""), f3(proto["seeds"][k]["mask_map50"]), f3(proto["seeds"][k]["macro_f1"]["plain"]), f3(proto["selection_score"][k])] for k in ("42", "1", "2")],
    "표 3. 대표 시드 선택(검증셋만 사용, 2026-11-05 protocol.json에 고정)")
seg_tbl = tbl(["", "전체", *CL], [
    ["mask mAP50", f3(s["test_mask_map50"]), *[f3(rr["per_class"][c]["map50"]) for c in CL]],
    ["mask mAP50-95", f3(s["test_mask_map50_95"]), *[f3(rr["per_class"][c]["map50_95"]) for c in CL]],
    ["논문 mask mAP50", "0.443", "–", "–", "–"]],
    f"표 4. 테스트셋 세그멘테이션 성능. 전체는 11/5 공식 평가, 클래스별은 보고용 재실행 값(전체 차이 {rr['diff_vs_1105']['mask_map50']:+.4f})")
val_rows = []
for c in CL:
    v = V[c]; plateau = [r["thr"] for r in curve if r["cls"] == c and abs(float(r["f1"]) - v["f1"]) < 1e-9]
    val_rows.append([c, THR[c], f3(v["f1"]), v["tp"], v["fp"], v["fn"], v["tn"], f"{plateau[0]}~{plateau[-1]}" if len(plateau) > 1 else "단일 정점"])
thr_tbl = tbl(["클래스", "선택 임계값", "검증 F1", "TP", "FP", "FN", "TN", "최고 F1 구간"], val_rows,
              f"표 5. 검증셋 40장에서 고른 임계값. 같은 최고 F1이 여러 값이면 가장 낮은 값을 골라 recall을 우선했다(검증 macro F1 {f3(s['val_macro_f1'])})")
onoff_tbl = tbl(["클래스", "임계값", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"],
                [[c, THR[c], T[c]["tp"], T[c]["fp"], T[c]["fn"], T[c]["tn"], f3(T[c]["precision"]), f3(T[c]["recall"]), f"<b>{f3(T[c]['f1'])}</b>"] for c in CL] +
                [["macro F1", "", "", "", "", "", "", "", f"<b>{f3(s['test_macro_f1'])}</b>"]],
                "표 6. 테스트셋 ON/OFF 결과(검증셋 임계값을 바꾸지 않고 적용). 논문 macro F1 0.896")
rob_tbl = tbl(["", "mask mAP50", "mask mAP50-95", "macro F1"], [
    ["제출 모델(시드 1)", f3(done["seeds"]["1"]["mask_map50"]), f3(done["seeds"]["1"]["mask_map50_95"]), f3(done["seeds"]["1"]["onoff"]["macro_f1"])],
    ["시드 3개 평균 ± 표준편차", f"{sm['mask_map50']['mean']:.3f} ± {sm['mask_map50']['std']:.3f}", f"{sm['mask_map50_95']['mean']:.3f} ± {sm['mask_map50_95']['std']:.3f}", f"{sm['macro_f1']['mean']:.3f} ± {sm['macro_f1']['std']:.3f}"],
    ["시드 3개, 공통 임계값 1개", "–", "–", f"{sm['macro_f1_common']['mean']:.3f} ± {sm['macro_f1_common']['std']:.3f}"],
    [f"train에 없는 촬영본만({uv['images']}장, 시드 1)", f3(uv["mask_map50"]), f3(uv["mask_map50_95"]), f"{uv['macro_f1_present']:.3f}"],
    ["논문 조건 재현(누수 포함, 참고)", f3(paper_run["mask_map50"]), f3(paper_run["mask_map50_95"]), f3(paper_run["onoff"]["macro"]["f1"])],
    ["논문", "0.443", "0.222", "0.896"]],
    "표 7. 결과가 얼마나 안정적인지 보는 비교(모두 테스트셋). 제출 모델 선택에는 쓰지 않았다")

ERR_TXT = {
    ("efflorescence", "FN", "URT007_2022_01_11_001_14386"): "흙손 자국이 있는 밝은 면 위의 작은 흰 덩어리. 주변과 밝기 차가 작고 크기가 작아 예측이 하나도 나오지 않았다(질감 대비 부족, 소형 객체).",
    ("efflorescence", "FN", "URT007_2022_01_11_001_1914"): "거푸집 줄무늬가 있는 회색 벽의 흐릿한 흰 영역. 경계가 퍼져 있고(확산적 경계) 밝은 변색과 구분이 어렵다. 모델의 가장 높은 백화 예측(0.002)은 오른쪽 이음부 가장자리에 있었다.",
    ("efflorescence", "FN", "URT007_2022_01_01_001_4832"): "표면이 떨어져 거칠어진 자리에 흰 석출물이 붙은 영상. 모델은 같은 자리를 spall_detach(0.24)로 보았고 백화는 0.001에 그쳤다. 두 결함이 겹친 영역에서의 클래스 혼동이다.",
    ("spall_detach", "FP", "URT007_2022_01_01_001_4832"): "위와 같은 영상. 정답 라벨에는 백화만 있지만 박락 질감(거친 면, 부서진 가장자리)이 실제로 보인다. 제공 라벨이 영상당 한 클래스만 있어 생긴 FP일 가능성이 있다(주석 모호성).",
    ("efflorescence", "FP", "URT017_2020_02_08_001_24512"): "점검자가 분필로 쓴 흰 글씨와 X 표시를 백화(0.80)로 예측했다(파란 선). 정답(분홍)을 둘러싼 흰 직선도 점검자의 분필 표시다. 흰 획이 주변과 대비가 커 백화의 시각 특징과 겹친다. 이 영상은 train에서 같은 원본 55장을 뺀 프레임이라 모델이 이 장면을 학습하지 않았다.",
    ("crack", "근접_OFF", "URT007_2022_01_13_001_14472"): "정답 OFF 영상 중 crack 점수가 가장 높은 사례(0.097, 임계값 0.1). 백화 영상의 가는 긁힘·흙손 자국에 균열 예측이 붙었다. 0.003 차이로 FP를 피했다.",
    ("crack", "근접_ON", "URT005_2021_01_03_001_13040"): "정답 ON 영상 중 crack 점수가 가장 낮은 사례(0.257). 그늘진 벽면 쪽 분기 균열이며, 임계값과 여유가 0.16 남아 있다.",
}
cards = []
for e in s["errors"] + s["near"]:
    key = (e["cls"], e["kind"], next(k[2] for k in ERR_TXT if k[0] == e["cls"] and k[1] == e["kind"] and e["stem"].startswith(k[2])))
    img = f"{W3}/error_cases/{e['cls']}_{e['kind']}_{e['stem'][:40]}.jpg"
    cards.append(f"""<div class="card"><img src="{b64(img)}"><div class="ct"><b>{e['cls']} {e['kind'].replace('_', ' ')}</b> · 정답 {e['gt']} · 점수 {e['conf']:.3f} / 임계값 {e['thr']}<br><code class="sm">{H.escape(e['stem'][:40])}</code><br>{ERR_TXT[key]}</div></div>""")
err_tbl = tbl(["클래스", "FP", "FN", "오류 영상의 최고 점수 / 임계값", "주된 원인"], [
    ["crack", 0, 0, "–", "오류 없음. 정답 OFF 최고 0.097로 임계값 0.1과 0.003 차이"],
    ["efflorescence", 1, 3, "FN 0.000·0.001·0.002 / 0.06, FP 0.803", "FN: 소형·저대비·확산 경계, 박락과 겹침. FP: 분필 표시"],
    ["spall_detach", 1, 0, "FP 0.241 / 0.01", "백화 영상의 박락 질감(정답에 없는 결함일 가능성)"]], "표 8. 클래스별 오류 요약")

GUIDE_CHANGES = [
    ["제외 대상", "분필 낙서", "분필·페인트 점검 표시(글씨, X·O 표시, 측정선)를 명시하고 백화와 혼동하지 않도록 경고", "테스트 FP: 분필 표시를 백화 0.80으로 예측"],
    ["efflorescence 포함", "주변보다 뚜렷이 희고 쌓인 질감", "작게 떨어진 덩어리와 경계가 흐린 면도 흰 석출물이면 포함. 단순히 밝은 변색은 제외", "테스트 FN 2건(소형, 흐린 경계)이 제공 라벨에서는 백화"],
    ["겹친 결함", "여러 결함이 겹치면 각 클래스로 각각", "박락면 위의 백화는 두 클래스를 모두 그린다(박락은 파인 영역, 백화는 흰 석출물 영역). 제공 라벨이 한 클래스뿐인 영상은 오류 분석에서 따로 표시", "테스트 FN·FP가 같은 영상(URT007_..._4832)에서 발생"],
    ["crack 제외", "(없음)", "틈이 보이지 않는 평행 긁힘·흙손 자국은 균열이 아니다", "정답 OFF 영상의 긁힘에 crack 0.097"],
    ["도색 띠 위 균열", "(없음)", "도색 띠 위에서도 균열선이 보이면 포함하고, 보이지 않는 구간은 끊어 별개 폴리곤", "2주차 직접 주석 3번 영상"],
    ["가린 물체", "배관 등이 가리면 빼고 양쪽 별개", "철선·고정 클립처럼 가는 물체에도 같은 규칙. 흐린 띠는 물체면 가린 물체, 그림자면 아래 결함을 포함", "2주차 직접 주석 6·7번 영상"],
    ["판단 보류", "임의로 라벨링하지 않고 기록", "작고 둥근 구멍(거푸집 타이 구멍일 수 있음)을 판단 보류 예시로 추가", "2주차 직접 주석 7번 영상"],
    ["도구 사용(신설)", "(없음)", "AI 보조(SAM) 사용 시 라벨을 먼저 고르고, 결함 하나에 맞는 작은 상자와 단일 객체 모드로 후보를 만든 뒤, 객체 목록에서 클래스와 불필요한 후보를 확인한다. 내보낸 뒤 겹쳐 그린 그림으로 클래스를 다시 점검한다", "2주차: 기본 라벨이 남아 균열 20개가 spall_detach로 저장됨, 큰 상자에서 후보 7개"],
]


def guide_html(standalone):
    head = "<h2>부록. 주석 기준서 수정본 (제출물 ②)</h2>" if not standalone else "<h1>주석 기준서 (수정본)</h1><p class='sub'>Step 3 지하 콘크리트 결함 · 22114553 전진성 · 1주차 기준서를 2주차 직접 주석과 3주차 오류 분석으로 고쳤다. <span class='new'>밑줄</span>이 바뀐 부분이다.</p>"
    return f"""<div class="guide">{head}
<h3>① 클래스 정의</h3>
{tbl(["클래스", "정의"], [["crack (0)", "콘크리트 표면이 갈라져 생긴 가늘고 불규칙한 선형 결함"], ["efflorescence (1)", "표면에 석출되어 쌓인 흰색·회백색의 불규칙한 물질"], ["spall_detach (2)", "표면이 떨어져 나가거나 들떠 단차·파임이 생긴 영역"]])}
<h3>② 포함 기준</h3>
<ul><li><b>crack</b>: 경로가 구불거리거나 분기하며 틈이 보일 것. 보수재 위에서도 균열선이 보이면 포함한다. <span class="new">도색 띠 위에서도 균열선이 보이면 포함한다.</span></li>
<li><b>efflorescence</b>: 주변보다 희고 표면에 물질이 쌓인 질감이 있을 것. 줄기처럼 흘러내린 부분도 포함한다. <span class="new">작게 떨어진 덩어리와 경계가 흐린 면도 흰 석출물이면 포함한다. 단순히 밝게 변색된 면은 제외한다.</span></li>
<li><b>spall_detach</b>: 깊이를 나타내는 그림자, 부서진 가장자리, 드러난 내부 골재 중 하나 이상이 있을 것. 금이 가고 들뜬 부분도 포함한다.</li></ul>
<h3>③ 제외 대상 (라벨링하지 않음)</h3>
<ul><li>파손 없는 시공 이음부와 거푸집 자국, 케이블·배관·철선·금속 덮개, 타일, 도색 띠</li>
<li>그림자, 금속·젖은 면의 반사, 습윤 얼룩, 먼지와 변색</li>
<li><span class="new">분필·페인트 점검 표시(글씨, X·O 표시, 측정선). 흰 획이라 백화로 착각하기 쉽다.</span></li>
<li><span class="new">틈이 보이지 않는 평행 긁힘과 흙손 자국(균열 아님)</span>, 균열선이 보이지 않는 보수재 띠</li></ul>
<h3>④ 경계 규칙</h3>
<ul><li><b>crack</b>: 균열선 양쪽에 약간의 여유를 둔 좁은 띠. 끊어져 보이는 구간은 별개 폴리곤으로 나누고, 끊겨 있지 않으면 분기까지 한 폴리곤으로 그린다. <span class="new">도색 띠 위에서 선이 안 보이는 구간은 끊어 별개로 그린다.</span></li>
<li><b>efflorescence</b>: 흰 물질이 끝나는 바깥 경계를 따른다. 떨어진 덩어리는 따로 그리고, 그 주변의 얼룩은 넣지 않는다.</li>
<li><b>spall_detach</b>: 파임과 부서진 가장자리까지 포함하고, 들뜬 판은 금이 간 선까지 포함한다. 주변에 번진 거뭇한 얼룩은 넣지 않는다.</li>
<li><b>공통</b>: 배관 등이 결함을 가리면 가린 물체를 빼고 양쪽을 별개 폴리곤으로 그린다. <span class="new">철선·고정 클립처럼 가는 물체에도 같은 규칙을 쓴다. 결함을 가로지르는 흐린 띠는 앞에 놓인 물체면 가린 물체로 빼고, 그림자면 그 아래 결함을 포함한다.</span> 결함이 화면 밖으로 이어지면 영상 가장자리까지 그린다. 여러 결함이 겹치면 각 클래스로 각각 그린다. <span class="new">박락면 위의 백화는 박락(파인 영역)과 백화(흰 석출물 영역)를 모두 그린다.</span></li></ul>
<h3>⑤ 판단 보류</h3>
<p>임의로 라벨링하지 않고 <code>results/week1/hold_list.csv</code>에 영상 이름과 이유를 적어 재검토한다. <span class="new">예: 작고 둥근 구멍(거푸집 타이 구멍인지 박락인지 불명), 영상 가장자리에 조금만 걸친 어두운 부분.</span></p>
<h3>⑥ 도구 사용 <span class="new">(신설)</span></h3>
<ul><li><span class="new">CVAT AI 보조(Segment Anything)를 쓸 때는 패널에서 라벨을 먼저 고르고, 결함 하나에 맞는 작은 상자와 단일 객체 모드로 후보를 만든다. 상자가 크면 비슷한 얼룩까지 여러 후보가 잡힌다.</span></li>
<li><span class="new">Done 뒤 객체 목록에서 클래스와 불필요한 후보를 확인하고, 놓친 구간은 폴리곤 도구로 직접 그린다. 결정은 주석자가 한다.</span></li>
<li><span class="new">내보낸 뒤 라벨을 영상에 겹쳐 그려 클래스를 다시 점검한다. 고친 내역은 <code>annotation/label_fixes.csv</code>에 남긴다.</span></li></ul>
<h3>⑦ 수정 내역과 근거</h3>
{tbl(["항목", "1주차", "수정본", "근거"], GUIDE_CHANGES)}
<figure style="margin:2pt 0"><img src="{b64('results/week1/class_examples.png')}" style="width:100%"><figcaption>클래스별 예시(색 윤곽선이 제공 라벨). 제외 대상과 오류 사례 그림은 최종 리포트 5절 참고</figcaption></figure>
</div>"""


CSS = """@page { size: A4; margin: 19mm 17mm 17mm 17mm; }
body { font-family: "Noto Serif KR", "Batang", "바탕", serif; font-size: 10pt; line-height: 1.58; color: #000; margin: 0; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6pt; } h2 { font-size: 12.5pt; margin: 14pt 0 5pt; border-bottom: 1px solid #000; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 8pt 0 3pt; break-after: avoid; }
p { margin: 3pt 0 5pt; } table { border-collapse: collapse; width: 100%; font-size: 8.7pt; } th, td { border: 1px solid #555; padding: 2.4pt 4pt; vertical-align: middle; word-break: keep-all; } th { background: #eee; }
.tbl { break-inside: avoid; margin: 6pt 0; } tr { break-inside: avoid; } .cap { font-size: 8.6pt; text-align: center; margin: 3pt 0 6pt; }
figure { margin: 6pt 0; text-align: center; break-inside: avoid; } figure img { max-width: 100%; } figcaption { font-size: 8.6pt; margin-top: 2pt; }
code { font-family: "D2Coding", "Consolas", monospace; font-size: 8.6pt; } code.sm { font-size: 7.3pt; word-break: break-all; }
pre { font-family: "D2Coding", "Consolas", monospace; font-size: 8pt; line-height: 1.33; border: 1px solid #999; padding: 5pt; white-space: pre-wrap; margin: 4pt 0; break-inside: avoid; }
mark { background: #ffe58a; } .hdr th { width: 14%; } ul { margin: 2pt 0 5pt; padding-left: 15pt; } li { margin: 1pt 0; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 6pt; } .card { border: 1px solid #999; break-inside: avoid; font-size: 8.4pt; line-height: 1.4; } .card img { width: 100%; display: block; } .ct { padding: 3pt 4pt; }
.guide { font-size: 9.2pt; line-height: 1.45; } .guide h3 { margin: 5pt 0 1pt; font-size: 9.6pt; } .guide p { margin: 1pt 0 3pt; } .guide table { font-size: 8.2pt; } .guide li { margin: 0; }
.new { text-decoration: underline; text-decoration-color: #c2410c; text-underline-offset: 2pt; } .sub { text-align: center; font-size: 9pt; color: #333; }
.pb { break-before: page; }"""

today = datetime.date.today()
hdr = f"""<table class="hdr"><tr><th>학 과</th><td>컴퓨터소프트웨어학부</td><th>학 번</th><td>22114553</td></tr><tr><th>이 름</th><td>전진성</td><th>제출일</th><td>{today.year}년 {today.month}월 {today.day}일</td></tr></table>"""
ef = T["efflorescence"]
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 최종 리포트</title><style>{CSS}</style></head><body>
<h1>Step 3 최종 리포트: 지하 콘크리트 결함 세그멘테이션과 영상 단위 선별</h1>
{hdr}

<h2>요약</h2>
<p>AI Hub 지하 콘크리트 영상에서 균열(crack), 백화(efflorescence), 박리·박락(spall_detach)을 YOLOv8s-Seg로 분할하고, 영상마다 결함 유무를 판정(ON/OFF)했다. 원본 단위로 누수를 없앤 train {ig['train']['kept']:,}장으로 시드 3개를 학습하고, <b>검증셋만으로</b> 대표 모델(시드 1)과 클래스별 임계값(crack {THR['crack']}, efflorescence {THR['efflorescence']}, spall_detach {THR['spall_detach']})을 정해 테스트 전에 파일로 고정했다. 테스트셋 40장에 그대로 적용한 결과 <b>mask mAP50 {f3(s['test_mask_map50'])}, mAP50-95 {f3(s['test_mask_map50_95'])}, ON/OFF macro F1 {f3(s['test_macro_f1'])}</b>이다(논문 0.443, 0.222, 0.896). mAP는 시드 3개 모두 논문보다 높았지만, macro F1의 차이는 테스트 1장이 바꾸는 폭과 비슷해 우열을 말하기 어렵다. 오류는 영상 단위로 5건이며, 그중 4건이 백화에서 나왔다. 원인은 작거나 흐린 백화, 박락과 겹친 백화, 분필 점검 표시였고, 이를 반영해 주석 기준서를 고쳤다.</p>

<h2>1. 문제 정의</h2>
<ul><li><b>결함 정의</b>: crack은 표면이 갈라진 가늘고 불규칙한 선, efflorescence는 표면에 석출되어 쌓인 흰 물질, spall_detach는 표면이 떨어지거나 들떠 파임·단차가 생긴 영역이다(부록 기준서 ①).</li>
<li><b>세그멘테이션을 고른 이유</b>: 균열은 폭이 수십 픽셀인 선이라 박스로 잡으면 박스 안 대부분이 배경이 된다. 백화와 박락은 모양이 불규칙해 영역 자체를 알아야 범위를 판단할 수 있다. 그래서 픽셀 단위 마스크를 예측한다.</li>
<li><b>평가 목표</b>: 두 개의 자로 잰다. mask mAP는 경계가 얼마나 맞는지를, ON/OFF는 결함이 있는 영상을 얼마나 잘 골라내는지를 잰다. 현장 선별에는 후자가 중요하다. 이 과제의 핵심 규칙은 <b>임계값과 모델을 검증셋에서 정하고 테스트셋에는 바꾸지 않고 적용하는 것</b>이다.</li></ul>

<h2>2. 데이터</h2>
<p><b>주석 기준</b>: 1주차 기준서를 2주차 직접 주석(15장, {ann['instances_total']}개)과 3주차 오류 분석으로 고쳐 부록에 실었다. 바뀐 점은 분필 표시 제외, 작거나 흐린 백화 포함, 박락면 위 백화를 두 클래스로 그리기, 긁힘 제외, AI 보조 도구 사용 규칙이다.</p>
<p><b>규모와 구성</b>: 제공 데이터는 train {ig['train']['files']:,} / val {ig['val']['files']} / test {ig['test']['files']}장이다. train은 원본 {ig['train']['origins']}프레임을 오프라인 증강한 것이고, 각 영상에는 한 클래스만 라벨되어 있다. 그래서 영상에 보이는 다른 결함은 정답에 없다(5절의 FP 해석에 영향).</p>
{data_tbl}
<p><b>원본 단위 분할</b>: 제공 분할(train/val/test)은 유지했다. 대신 읽을 수 없는 0바이트 영상 63장과, test 프레임 <code class="sm">URT017_2020_02_08_001_24512</code>와 원본이 같은 train 영상 55장을 train에서 뺐다. 그 결과 원본 프레임 기준 겹침은 0이다. 촬영본(영상 묶음) 단위로는 test 40장 중 {40 - uv['images']}장이 train과 같은 촬영본에서 나왔다. 그래서 train에 없는 촬영본 {uv['images']}장만으로 따로 평가해 표 7에 함께 적었다.</p>

<h2>3. 방법</h2>
{set_tbl}
<p><b>모델 선택</b>: 시드 42·1·2를 같은 설정으로 학습했다. 검증셋에서 (mask mAP50 + macro F1)/2가 가장 높은 시드 1을 제출 모델로 정했다(표 3). 이 규칙과 결과는 테스트를 돌리기 전인 2026-11-05에 <code>results/improve/protocol.json</code>(sha256 {done['protocol_sha256'][:12]}…)에 기록했다. 테스트 단계 코드는 이 파일을 읽기만 한다. 타일 추론도 후보였지만 검증 macro F1이 낮아(0.860 &lt; 0.868) 쓰지 않았다. 2주차에 제출한 강의 설정 모델(patience 20, 36 epoch에서 조기 종료)은 테스트에 평가하지 않았고, 비교나 선택에도 쓰지 않았다.</p>
{sel_tbl}
<p><b>ON/OFF 판정</b>: 영상에 그 클래스 정답 인스턴스가 하나라도 있으면 정답 ON이다. 그 클래스 예측 중 confidence가 임계값 이상인 것이 하나라도 있으면 예측 ON이다. 영상마다 클래스별 최고 confidence를 한 번 구해 두고, 임계값별 TP·FP·FN·TN을 세어 클래스별 F1과 macro F1을 계산했다. 강의 코드의 <code>predict</code>는 기본 conf 0.25로 예측을 먼저 걸러, 그대로 쓰면 0.25보다 낮은 임계값이 모두 같은 결과가 된다. 그래서 conf 0.001로 예측을 모두 받은 뒤 임계값을 적용했다(코드 1).</p>
<pre>{code_onoff}</pre><p class="cap">코드 1. src/improve_eval.py의 ON/OFF 판정 부분</p>
<p><b>임계값 탐색</b>: 검증셋에서만 강의 범위 8개 값(0.001~0.3)에 0.4~0.7을 더해 탐색했다. 클래스별 F1이 가장 높은 값을 고르고, 동점이면 낮은 값을 골랐다. 범위를 넓힌 것은 앞서 해 본 논문 조건 실험(5절 표 7 마지막 행)의 검증 곡선이 0.3까지 평평했기 때문이다. 그 실험의 테스트를 본 뒤에 정한 것이라 강의 범위로 고른 결과도 함께 기록했다. 제출 모델은 세 클래스 모두 같은 값이 나왔다(시드 2의 spall_detach만 0.01 → 0.5로 달라짐). 코드 2의 표시한 줄이 검증셋에서 고르고(위) 테스트에서 읽기만 하는(아래) 부분이다.</p>
<pre>{code_thr}</pre><p class="cap">코드 2. 검증셋 탐색 → protocol.json 고정(한 번만 쓰기, "x" 모드) → 테스트 적용</p>

<h2>4. 결과</h2>
<h3>4.1 세그멘테이션 성능 (제출물 ③)</h3>
{seg_tbl}
<p>spall_detach가 가장 높고 crack이 가장 낮다. 균열은 폭이 960 입력에서 중앙값 {cw['px_at_960']['q50']} px라, 경계가 1~2 px만 어긋나도 IoU가 크게 떨어진다. 그래서 mAP50-95가 {f3(rr['per_class']['crack']['map50_95'])}에 그친다. 백화는 경계가 퍼져 있어 주석 범위 자체가 흔들린다. 박락은 파임과 가장자리가 뚜렷해 IoU 0.5 기준을 넘기 쉽다.</p>
<h3>4.2 검증셋 임계값과 F1 곡선 (제출물 ④)</h3>
<figure><img src="{b64(f'{W3}/val_threshold_f1.png')}"><figcaption>그림 1. 검증셋 클래스별 F1 곡선(점선이 선택한 임계값, 가로축은 로그)</figcaption></figure>
{thr_tbl}
<p>세 클래스의 곡선 모양이 다르다. crack은 0.1에서 뾰족한 정점이 있어, 그보다 낮추면 긁힘 등의 FP가, 높이면 FN이 늘어난다. 백화는 0.06~0.5에서 평평하고, 박락은 0.01~0.1에서 평평해 가장 낮은 값을 골랐다. 정점 위치가 클래스마다 달라 클래스별 임계값을 썼다. 다만 이 모델은 검증셋에서 공통 임계값 0.1로도 macro F1이 같았다({f3(proto['seeds']['1']['macro_f1']['plain_common'])}). 테스트 시드 평균에서도 공통 임계값이 조금 높았다(표 7). 이 데이터 크기에서는 클래스별 임계값의 이득이 뚜렷하지 않다.</p>
<h3>4.3 테스트셋 ON/OFF 결과 (제출물 ⑤)</h3>
{onoff_tbl}
<p>crack은 13장을 모두 맞혔고, 박락은 FP 1장, 백화는 FN 3장과 FP 1장이 틀렸다. 백화 recall이 {f3(ef['recall'])}로 가장 약하다. 테스트 결과를 본 뒤 임계값을 바꾸지 않았다.</p>
<h3>4.4 결과의 안정성</h3>
{rob_tbl}
<p>mask mAP는 시드 3개 모두 논문보다 0.2 이상 높다. 누수를 그대로 둔 논문 조건 재현도 0.631로 논문보다 높아, 차이가 누수 제거에서 온 것은 아니다. 원인(평가 코드·환경 차이 등)은 확인하지 못했다. macro F1은 시드에 따라 0.877~0.933으로, 차이가 테스트 1장이 클래스 F1을 바꾸는 폭(약 0.03~0.04)과 비슷하다. 그래서 논문과 같은 수준으로 판단한다. train에 없는 촬영본 {uv['images']}장에서도 mAP50 {f3(uv['mask_map50'])}로 논문보다 높았다. 다만 이 부분 집합의 crack mAP50은 {f3(uv['per_class_mask_map50']['crack'])}로 낮아, 처음 보는 현장의 균열 경계는 여전히 어렵다.</p>

<h2 class="pb">5. 오류 분석 (제출물 ⑥)</h2>
<p>임계값을 고정한 뒤 테스트셋에서 클래스별 FP·FN 영상을 모두 뽑았다(5건). crack은 오류가 없어, 임계값에 가장 가까웠던 사례 2건을 함께 보았다. 그림의 분홍 선은 정답이고(벽의 분필 표시가 흰색이라 구분), 색 선은 임계값 이상 예측(빨강 crack, 파랑 efflorescence, 초록 spall_detach)이고, 노란 선은 임계값 아래의 최고 점수 예측이다.</p>
{err_tbl}
<div class="cards">{''.join(cards)}</div>
<h3>결함별 오류 특성</h3>
<ul><li><b>efflorescence</b>: FN 3건의 백화 최고 점수가 모두 0.002 이하다. 임계값을 탐색 범위의 최소값 0.001까지 내려야 두 건이 겨우 ON이 되고 한 건은 끝까지 예측이 없다. 그런데 검증셋에서는 0.001의 백화 F1이 0.848로 선택값(0.867)보다 낮다. 즉 임계값으로 고칠 수 없는 <b>표현 부족</b>이다. 소형·저대비 백화와 박락과 겹친 백화가 학습 데이터에서 드문 것이 원인으로 보인다.</li>
<li><b>오탐 요인</b>: 분필 점검 표시가 백화로, 박락 질감이 있는 백화 영상이 박락으로 판정되었다. 앞의 것은 기준서의 제외 대상이 실제로 혼동을 일으킨 사례다. 뒤의 것은 영상당 한 클래스만 라벨된 데이터 특성 때문에 실제 결함이 FP로 셈해졌을 수 있다.</li>
<li><b>crack</b>: 영상 단위 오류는 없지만, 정답 OFF 영상의 긁힘이 0.097로 임계값 0.1과 0.003 차이였다. 박락도 정답 ON 영상 하나가 0.019로 임계값 0.01을 겨우 넘었다. 테스트가 커지면 이런 경계 사례가 오류로 바뀔 수 있다.</li>
<li><b>개선 방향</b>: 분필 표시를 배경으로 넣은 하드 네거티브, 작은 백화와 박락면 위 백화의 추가 주석(두 클래스 모두), 긁힘 하드 네거티브. 기준서에는 이미 반영했다(부록 ⑦).</li></ul>

<h2>6. 한계</h2>
<ul><li><b>테스트 규모</b>: 테스트는 40장, 클래스당 정답 ON 13~14장이다. 1장이 클래스 F1을 약 0.03~0.04 바꾸므로, 논문과의 F1 차이나 클래스별 임계값의 효과는 이 해상도 안에 있다.</li>
<li><b>촬영본 겹침</b>: 테스트 {40 - uv['images']}장은 train과 같은 촬영본(다른 프레임)이다. 처음 보는 현장 성능은 {uv['images']}장 부분 집합으로만 볼 수 있다.</li>
<li><b>라벨 특성</b>: 영상당 한 클래스만 라벨되어 있어, 보이는 다른 결함이 FP로 셈해질 수 있다. 백화처럼 경계가 흐린 결함은 주석자마다 범위가 달라진다(2주차 직접 주석과 제공 라벨의 IoU 중앙값 백화 {ann['iou_with_provided']['efflorescence']['median']:.2f}).</li>
<li><b>물리 단위 정량화 불가</b>: 촬영 거리와 축척 정보가 없어 균열 폭(mm)이나 면적(m²)으로 바꿀 수 없다. 결과는 픽셀 단위다.</li>
<li><b>단일 모델</b>: YOLOv8s-Seg 하나만 썼다. 시드 3개로 흔들림만 확인했고, 다른 구조나 앙상블은 비교하지 않았다.</li>
<li><b>보고용 재실행</b>: 클래스별 mAP50-95와 오류 그림을 만들려고 같은 모델로 테스트 추론을 두 번 더 했다(2026-11-18, 두 번째는 정답 선 색을 바꿔 그림만 다시 그리려고 오류 영상 7장만 추론. <code>test_attempts.log</code>에 기록). 임계값과 모델은 바꾸지 않았고, 전체 mAP50 차이는 {rr['diff_vs_1105']['mask_map50']:+.4f}로 GPU 차이 수준이다.</li></ul>

<h2>7. 재현 정보</h2>
<pre># 1) 데이터 정리(누수 제거) — 서버(RTX 4090)
python src/week2_prepare.py
# 2) 시드 3개 학습(60 epoch). 끊기면 다시 실행하면 이어서 학습
sh src/improve_train.sh
# 3) 검증셋만으로 임계값·추론 방식·대표 시드 결정 → results/improve/protocol.json (한 번만 쓰기)
python src/improve_eval.py --stage val
# 4) 고정값을 테스트셋에 적용(1회, 이미 결과가 있으면 실행 거부)
python src/improve_eval.py --stage test
# 5) 3주차 정리·리포트·제출 zip — PC(RTX 2070 SUPER). --report-pass는 클래스별 mAP50-95와 오류 그림용 테스트 추론
python src/week3_eval.py --report-pass
python main.py week3
제출 모델 best.pt sha256 {s['weights_sha256']}
protocol.json sha256 {done['protocol_sha256']}</pre>
<p>학습·평가는 RTX 4090 서버(Python 3.11.18, torch 2.14.1, ultralytics 8.4.198), 정리와 보고용 재실행은 RTX 2070 SUPER PC(Python 3.10.12, torch 2.14.1, ultralytics 8.4.198)에서 했다. 패키지 버전은 <code>requirements.lock.txt</code>에, 데이터 설정은 <code>dataset/data.yaml</code>(0 crack, 1 efflorescence, 2 spall_detach)에 있다. 학습 시드는 1(deterministic)이고, 직접 주석 묶음 선택 시드는 42다. 제출 모델 결정 기록은 <code>results/week3/model_decision.json</code>에 있다.</p>
{tbl(["강의 제출물", "산출물"], [["① 리포트", "이 문서"], ["② 주석 기준서(수정본)", "부록(이 문서 마지막 쪽)"], ["③ 세그멘테이션 성능", "4.1절 표 4, results/improve/test_results.json · results/week3/week3_summary.json"], ["④ 임계값 F1 곡선", "4.2절 그림 1·표 5, results/week3/val_threshold_f1.png · .csv, results/improve/protocol.json"], ["⑤ ON/OFF 결과표", "4.3절 표 6, results/week3/test_onoff.csv"], ["⑥ 오류 사례", "5절, results/week3/error_cases/ · error_cases.csv"], ["가중치", "best.pt (clean_s1)"]])}

<div class="pb">{guide_html(False)}</div>
</body></html>"""

guide = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>주석 기준서 수정본</title><style>{CSS}</style></head><body>{guide_html(True)}</body></html>"""

os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
outs = {"week3_report": (html, "Step3_최종리포트_22114553_전진성.pdf"), "week3_guideline": (guide, "Step3_주석기준서_수정본_22114553_전진성.pdf")}
for k, (doc, pdf) in outs.items():
    hp = os.path.join(ROOT, "report", f"{k}.html"); open(hp, "w", encoding="utf-8").write(doc); print("HTML", k, len(doc) // 1024, "KB")
    if a.pdf:
        os.makedirs(WIN_TMP, exist_ok=True); shutil.copy(hp, f"{WIN_TMP}/{k}.html")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", rf"--print-to-pdf=C:\Users\user\semg_report_tmp\{k}.pdf", "--virtual-time-budget=10000",
                        f"file:///C:/Users/user/semg_report_tmp/{k}.html"], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = os.path.join(ROOT, "report", pdf); shutil.copy(f"{WIN_TMP}/{k}.pdf", out); print("PDF", out, os.path.getsize(out) // 1024, "KB")
if a.pdf:
    shutil.rmtree(WIN_TMP, ignore_errors=True)
