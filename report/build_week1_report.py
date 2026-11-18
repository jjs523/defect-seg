"""results/week1/ 산출물로 Step 3 1주차 보고서를 만든다. 실행: python report/build_week1_report.py --pdf"""
import base64, collections, csv, html as H, json, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "week1"); OUT_DIR = os.path.join(ROOT, "report")
WIN_TMP = "/mnt/c/Users/user/semg_report_tmp"; CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
d = json.load(open(f"{R}/data_summary.json")); nd = json.load(open(f"{R}/near_duplicates_summary.json")); br = json.load(open(f"{R}/brightness.json"))
rows = list(csv.DictReader(open(f"{R}/data_index.csv")))
env = [l for l in open(f"{R}/env_check.txt").read().splitlines() if l.strip()]
hold = list(csv.DictReader(open(f"{R}/hold_list.csv")))
CL = ["crack", "efflorescence", "spall_detach"]

def img(n):
    mt = "jpeg" if n.endswith("jpg") else "png"
    return f"data:image/{mt};base64," + base64.b64encode(open(f"{R}/{n}", "rb").read()).decode()
def tbl(head, body, cap="", cls="tbl"):
    t = f"<div class='{cls}'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
    t += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>"
    return t + (f"<p class='cap'>{cap}</p>" if cap else "") + "</div>"

# 영상(동영상 단위) 공유: 프레임 번호를 뺀 이름이 같으면 같은 촬영본으로 본다
vid = collections.defaultdict(lambda: collections.defaultdict(set))
for r in rows:
    m = re.match(r"(.+)_(\d+)$", r["origin"])
    vid[r["split"]][m.group(1)].add(int(m.group(2)))
def shared(a, b):
    sh = set(vid[a]) & set(vid[b]); gaps = [min(abs(f - g) for g in vid[a][v]) for v in sh for f in vid[b][v]]
    return len(sh), len(vid[b]), sum(x <= 10 for x in gaps)
vt, vv = shared("train", "test"), shared("train", "val")
origin_cnt = {c: len({r["origin"] for r in rows if r["split"] == "train" and r[c] == "1"}) for c in CL}
s = d
readable = sum(s[k]["images"] for k in ["train", "val", "test"])
nread = {k: len([r for r in rows if r["split"] == k]) for k in ["train", "val", "test"]}
# 주의: images 값은 파일 수(0바이트 포함)
data_tbl = tbl(["항목", "train", "val", "test"], [
    ["jpg 파일 수", 4036, 40, 40], ["읽을 수 있는 영상", nread["train"], nread["val"], nread["test"]],
    ["해상도", "1920×1080 3,647 / 1080×1920 326", "1920×1080 40", "1920×1080 40"],
    ["원본 프레임 수", s["unique_origins"]["train"], s["unique_origins"]["val"], s["unique_origins"]["test"]],
    ["이름상 구성", f"원본 {s['train']['kinds']['plain']} / 증강 {s['train']['kinds']['aug']:,}", f"원본 {s['val']['kinds']['plain']} / Roboflow 내보내기 {s['val']['kinds']['roboflow']}", f"원본 {s['test']['kinds']['plain']} / Roboflow 내보내기 {s['test']['kinds']['roboflow']}"],
] + [[f"{c}: 영상 / 인스턴스", *[f"{s[k]['images_per_class'][c]:,} / {s[k]['instances_per_class'][c]:,}" for k in ["train", "val", "test"]]] for c in CL] + [
    ["빈 라벨(결함 없는 영상)", 0, 0, 0],
], "표 1. 데이터 구성. 클래스 집계는 읽을 수 있는 영상 기준. 형식 오류 폴리곤은 0개")

hn = [
 ("joint", "시공 이음부", "crack", "직선으로 길게 이어지는 어두운 틈이라 균열로 오인될 수 있다."),
 ("cable", "케이블·배관", "crack", "벽을 가로지르는 곧은 어두운 선이라 균열로 오인될 수 있다."),
 ("shadow", "그림자(로 보임)", "crack", "가장자리가 흐린 어두운 띠가 이음부 위까지 가로질러, 넓은 균열로 오인될 수 있다."),
 ("reflection", "금속 덮개 반사", "efflorescence", "얼룩무늬가 있는 밝은 금속면이라 흰 석출물인 백태로 오인될 수 있다."),
 ("repair", "보수 흔적(로 보임)", "crack", "구불구불한 녹색 띠가 균열 경로처럼 보여 균열로 오인될 수 있다."),
 ("stain", "분필 낙서·표면 표시", "efflorescence·crack", "흰 글씨와 흰 줄이 백태나 가는 균열로 오인될 수 있다."),
 ("stain2", "습윤 얼룩", "efflorescence·spall_detach", "경계가 불규칙한 짙은 얼룩이 라벨된 백태에 붙어 있어, 백태 범위에 묶이거나 박리로 오인될 수 있다."),
]
cfg = {h["key"]: h for h in json.load(open(os.path.join(ROOT, "src", "week1_examples.json")))["hard_negatives"]}
hn_tbl = tbl(["캡처", "종류", "혼동 대상", "왜 헷갈리는가", "영상"], [
    [f"<img src='{img('hn_' + k + '.jpg')}' style='height:17mm;max-width:38mm'>", n, c, t, f"<code class='sm'>{cfg[k]['stem']}</code>"] for k, n, c, t in hn], "표 2. 하드 네거티브 후보 7건. 원본 해상도 잘라내기")

hold_tbl = tbl(["영상", "내용", "이유"], [[f"<code class='sm'>{H.escape(h['stem'])}</code>", H.escape(h["what"]), H.escape(h["reason"])] for h in hold], "표 3. 판단 보류 목록 (results/week1/hold_list.csv)")

html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 1주차 보고서</title><style>
@page {{ size: A4; margin: 20mm 18mm 18mm 18mm; }}
body {{ font-family: "Noto Serif KR", "Batang", "바탕", serif; font-size: 10.2pt; line-height: 1.6; color: #000; margin: 0; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 8pt; }} h2 {{ font-size: 12.5pt; margin: 15pt 0 5pt; border-bottom: 1px solid #000; padding-bottom: 2pt; break-after: avoid; }}
h3 {{ font-size: 10.6pt; margin: 8pt 0 2pt; break-after: avoid; }} p {{ margin: 3pt 0 5pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.9pt; }} th, td {{ border: 1px solid #555; padding: 2.5pt 4pt; vertical-align: middle; word-break: keep-all; }} th {{ background: #eee; }}
.tbl {{ break-inside: avoid; margin: 6pt 0; }} tr {{ break-inside: avoid; }} .cap {{ font-size: 8.8pt; text-align: center; margin: 3pt 0 6pt; }}
figure {{ margin: 6pt 0; text-align: center; break-inside: avoid; }} figure img {{ max-width: 100%; }} figcaption {{ font-size: 8.8pt; margin-top: 2pt; }}
code {{ font-family: "D2Coding", "Consolas", monospace; font-size: 8.8pt; }} code.sm {{ font-size: 7pt; word-break: break-all; }} pre {{ font-family: "D2Coding", "Consolas", monospace; font-size: 8.3pt; line-height: 1.35; border: 1px solid #999; padding: 5pt; white-space: pre-wrap; margin: 4pt 0; }}
.hdr th {{ width: 14%; }} ul {{ margin: 2pt 0 5pt; padding-left: 15pt; }} li {{ margin: 1pt 0; }}
.guide {{ break-before: page; break-after: page; font-size: 9.1pt; line-height: 1.42; }} .guide h2 {{ margin-top: 0; }} .guide h3 {{ margin: 4.5pt 0 1pt; font-size: 9.5pt; }} .guide p {{ margin: 1pt 0 2.5pt; }} .guide table {{ font-size: 8.3pt; }} .guide ul {{ margin: 0 0 2pt; }} .guide li {{ margin: 0; }}
</style></head><body>
<h1>Step 3 1주차 과제 보고서</h1>
<table class="hdr"><tr><th>학 과</th><td>컴퓨터소프트웨어학부</td><th>학 번</th><td>22114553</td></tr><tr><th>이 름</th><td>전진성</td><th>제출일</th><td>2026년 11월 5일</td></tr></table>

<h2>요약</h2>
<p>YOLOv8s-Seg 실행 환경을 구축하고, 제공 데이터 4,116장을 확인했다. 수는 논문의 4,036 / 40 / 40과 같지만, 데이터를 열어 보니 <b>train 63장이 0바이트 파일</b>이었고, train은 <b>원본 120프레임을 증강해 만든 것</b>이며, <b>test 1장은 train 원본과 같은 프레임</b>이었다. 또 각 영상에는 한 클래스만 라벨되어 있어 <b>보이는 다른 결함이 라벨에 없는 경우</b>가 있다. 이 네 가지는 2·3주차 학습과 오류 분석에 영향을 주므로 6절에 정리했다. 관찰과 기준서 작성에는 train만 사용하고 test 영상은 보지 않았다.</p>

<h2>1. 환경 확인 (제출물 ①)</h2>
<pre>{H.escape(chr(10).join(env))}</pre>
<p>강의 코드대로 <code>model loaded</code>와 <code>cuda: True</code>를 확인했다. 패키지 버전은 <code>requirements.lock.txt</code>에 고정했고 <code>setup.sh</code>로 같은 환경을 다시 만들 수 있다.</p>

<h2>2. 데이터 요약 (제출물 ②)</h2>
<p>강의의 <code>glob("data/images/**/*.jpg", recursive=True)</code>는 4,116장을 찾는다. 다만 0바이트 파일 때문에 <code>cv2.imread</code>가 None을 돌려줄 수 있어, 강의의 샘플 코드는 이 파일을 뽑으면 멈춘다. 그래서 읽을 수 있는 영상만 사용했다.</p>
{data_tbl}
<p>train 원본 프레임은 crack {origin_cnt['crack']}, efflorescence {origin_cnt['efflorescence']}, spall_detach {origin_cnt['spall_detach']}개다. 영상 수로는 spall_detach가 많지만(2,372장), 원본이 많아서이고, 원본당 증강 배수는 크게 다르지 않다(crack 약 {716/origin_cnt['crack']:.0f}배, efflorescence 약 {885/origin_cnt['efflorescence']:.0f}배, spall_detach 약 {2372/origin_cnt['spall_detach']:.0f}배). 인스턴스의 면적 중앙값은 train 기준 영상의 {min(s['train']['median_area_pct'].values()):.1f}~{max(s['train']['median_area_pct'].values()):.1f} %로 작다.</p>
<p>train 원본 {br['n']}장의 평균 밝기는 중앙값 {br['mean_q10_50_90'][1]} (10~90 % 구간 {br['mean_q10_50_90'][0]}~{br['mean_q10_50_90'][2]}, 범위 {br['mean_min_max'][0]}~{br['mean_min_max'][1]})이고, 표준편차는 중앙값 {br['std_q10_50_90'][1]}다. 영상 간 조도 차이가 분명히 있다.</p>

<h2>3. 샘플 관찰 (제출물 ③)</h2>
<figure><img src="{img('samples9.png')}" style="width:74%"><figcaption>그림 1. samples9.png: train 원본에서 클래스별 3장(시드 42). 윤곽선 없는 원본</figcaption></figure>
<ul><li><b>crack</b>: 구불거리고 분기하는 가는 선. 보수재로 보이는 띠 안에 있는 경우도 있다.</li>
<li><b>efflorescence</b>: 불규칙한 흰색·회백색 덩어리. 넓은 면부터 이음부 주변의 좁은 띠까지 크기 차이가 크다.</li>
<li><b>spall_detach</b>: 표면이 떨어져 나간 파임. 가로 이음부를 따라 자주 나타난다.</li>
<li><b>배경</b>: 이음부, 금속 덮개, 케이블, 타일, 도색 띠, 낙서가 흔하다. 원본 관찰 결과 어두운 영상보다 얼룩과 표면 무늬의 혼동이 더 눈에 띄었다.</li></ul>

<h2>4. 하드 네거티브 (제출물 ④)</h2>
<p>강의의 여섯 종류를 train 원본 71장을 눈으로 확인해 모두 찾았다. 그림 2의 빨간 상자가 혼동 요소이고 색 윤곽선이 제공 라벨이다.</p>
{hn_tbl}
<figure><img src="{img('hard_negatives.png')}" style="width:100%"><figcaption>그림 2. 하드 네거티브와 라벨 위치. 마지막 두 칸은 기준이 흔들리는 사례: 녹색 띠가 crack으로 라벨된 경우와, 뚜렷한 균열이 라벨되지 않은 경우</figcaption></figure>

<div class="guide">
<h2>5. 주석 기준서 (제출물 ⑤)</h2>
<p>제공 라벨과 train 원본을 관찰해 정리했다. 2주차에 추가 주석을 만들면 이 기준을 따른다.</p>
<h3>① 클래스 정의</h3>
{tbl(["클래스", "정의"], [["crack (0)", "콘크리트 표면이 갈라져 생긴 가늘고 불규칙한 선형 결함"], ["efflorescence (1)", "표면에 석출되어 쌓인 흰색·회백색의 불규칙한 물질"], ["spall_detach (2)", "표면이 떨어져 나가거나 들떠 단차·파임이 생긴 영역"]])}
<h3>② 포함 기준</h3>
<ul><li><b>crack</b>: 경로가 구불거리거나 분기하며 틈이 보일 것. 보수재 위에서도 균열선이 보이면 포함한다.</li>
<li><b>efflorescence</b>: 주변보다 뚜렷이 희고 표면에 물질이 쌓인 질감이 있을 것. 줄기처럼 흘러내린 부분도 포함한다.</li>
<li><b>spall_detach</b>: 깊이를 나타내는 그림자, 부서진 가장자리, 드러난 내부 골재 중 하나 이상이 있을 것. 금이 가고 들뜬 부분도 포함한다.</li></ul>
<h3>③ 제외 대상 (라벨링하지 않음)</h3>
<ul><li>파손 없는 시공 이음부와 거푸집 자국, 케이블·배관·금속 덮개, 타일, 도색 띠</li>
<li>그림자, 금속·젖은 면의 반사, 습윤 얼룩, 먼지와 변색, 분필 낙서</li>
<li>균열선이 보이지 않는 보수재 띠</li></ul>
<h3>④ 경계 규칙</h3>
<ul><li><b>crack</b>: 균열선 양쪽에 약간의 여유를 둔 좁은 띠. 끊어져 보이는 구간은 별개 폴리곤으로 나누고, 끊겨 있지 않으면 분기까지 한 폴리곤으로 그린다.</li>
<li><b>efflorescence</b>: 흰 물질이 끝나는 바깥 경계를 따른다. 떨어진 덩어리는 따로 그리고, 그 주변의 얼룩은 넣지 않는다.</li>
<li><b>spall_detach</b>: 파임과 부서진 가장자리까지 포함하고, 들뜬 판은 금이 간 선까지 포함한다.</li>
<li><b>공통</b>: 배관 등이 결함을 가리면 가린 물체를 빼고 양쪽을 별개 폴리곤으로 그린다. 결함이 화면 밖으로 이어지면 영상 가장자리까지 그린다. 여러 결함이 겹치면 각 클래스로 각각 그린다.</li></ul>
<h3>⑤ 판단 보류</h3>
<p>임의로 라벨링하지 않고 <code>results/week1/hold_list.csv</code>에 영상 이름과 이유를 적어 재검토한다. 현재 train 원본 중 눈으로 확인한 71장에서 {len(hold)}건을 분리했다(6절 표 3).</p>
<h3>⑥ 예시 영상</h3>
<figure style="margin:2pt 0"><img src="{img('class_examples.png')}" style="width:100%"><figcaption>그림 3. 클래스별 예시(색 윤곽선이 라벨). 제외 대상 예시는 그림 2와 표 2 참고</figcaption></figure>
</div>

<h2>6. 데이터에서 발견한 문제</h2>
<h3>(1) train의 0바이트 영상 63장</h3>
<p>zip 항목 자체의 크기가 0이고 CRC 검사는 통과하므로 다운로드나 압축 해제 오류가 아니라 원본 압축에 들어 있던 상태다. 라벨은 있고(crack 36, spall_detach 27), 모두 URT020의 원본 프레임 3개에서 나왔다. 함께 들어 있는 점검 CSV는 파일 존재만 확인해 이를 잡지 못했다. 해당 원본의 다른 파일이 남아 있어 프레임 자체가 사라지지는 않는다. 실제 train은 <b>3,973장</b>이다. 다음 주 학습에서는 ultralytics가 이 파일을 어떻게 처리하는지 로그로 확인한다. 목록: <code>results/week1/unreadable_images.csv</code>.</p>
<h3>(2) 분할 간 겹침</h3>
<p>test의 <code>URT017_2020_02_08_001_24512</code>는 train 원본과 같은 프레임이다(저해상도 상관 1.0). train에 이 프레임에서 나온 파일이 55개 있다. 이름이 정확히 같은 프레임은 이 1건이고, 다른 영상 중 상관이 0.95 이상인 경우는 없었다. 다만 같은 촬영본을 공유하는 경우는 많다. test 촬영본 {vt[1]}개 중 {vt[0]}개, val 촬영본 {vv[1]}개 중 {vv[0]}개가 train과 겹치고, 프레임 번호 차이가 10 이하인 경우가 test {vt[2]}장, val {vv[2]}장이다. 그래서 강의의 "원본 단위 분할" 기준으로 보면 test 성능은 낙관적일 수 있다. 채점 비교를 위해 test 구성은 바꾸지 않고, 2주차에 이 프레임들을 뺀 추가 평가를 할지 검토한다.</p>
<h3>(3) 영상당 한 클래스 라벨</h3>
<p>train·val·test의 모든 영상에 라벨된 클래스가 하나뿐이고 빈 라벨은 없다. 그런데 그림 2의 마지막 영상처럼 다른 결함이 눈에 보이는 경우가 있다. 따라서 3주차에 모델이 라벨 없는 진짜 결함을 맞혀도 FP로 집계될 수 있으므로, 오류 분석에서 이 유형을 따로 구분한다. 결함 없는 영상이 없다는 점도 ON/OFF 평가에서 함께 고려한다.</p>
<h3>(4) 그 밖의 점</h3>
<ul><li>증강본이 96 %를 넘으므로 train 장수는 다양성을 뜻하지 않는다. 실제 다양성은 원본 120프레임 수준이다. 좌우 반전 쌍도 있다.</li>
<li><code>data.yaml</code>의 <code>path</code>가 다른 PC 경로라 2주차에 프로젝트 경로로 고친다.</li>
<li>점검 CSV의 val·test 파일 목록은 현재 파일과 일부 다르다(이름 기준 val 13, test 6건). 그래서 집계는 실제 파일에서 다시 냈다. train은 0바이트 63장을 빼면 CSV와 일치한다.</li></ul>
{hold_tbl}

<h2>7. 제출물 대조</h2>
{tbl(["강의 제출물", "산출물"], [["① 환경 확인", "1절, results/week1/env_check.txt"], ["② 데이터 요약", "2절 표 1, results/week1/data_summary.json · data_index.csv"], ["③ 샘플 그림", "3절 그림 1, results/week1/samples9.png"], ["④ 하드 네거티브 3종 이상", "4절 표 2·그림 2 (7건), results/week1/hn_*.jpg"], ["⑤ 주석 기준서 1페이지", "5절"], ["실행", "<code>python main.py week1</code>"]])}
</body></html>"""
os.makedirs(OUT_DIR, exist_ok=True)
hp = os.path.join(OUT_DIR, "week1_report.html"); open(hp, "w", encoding="utf-8").write(html); print("HTML", len(html) // 1024, "KB")
if "--pdf" in sys.argv:
    os.makedirs(WIN_TMP, exist_ok=True); shutil.copy(hp, f"{WIN_TMP}/week1_report.html")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", r"--print-to-pdf=C:\Users\user\semg_report_tmp\week1_report.pdf", "--virtual-time-budget=10000", "file:///C:/Users/user/semg_report_tmp/week1_report.html"], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = os.path.join(OUT_DIR, "Step3_1주차과제_보고서_22114553_전진성.pdf"); shutil.copy(f"{WIN_TMP}/week1_report.pdf", out); shutil.rmtree(WIN_TMP); print("PDF", out, os.path.getsize(out) // 1024, "KB")
