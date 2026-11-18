"""단일 진입점: python main.py week1 | python main.py week2 [--annotation 내보낸파일.zip --tool CVAT] | python main.py week3
week3는 테스트를 다시 돌리지 않는다(results/improve의 고정 결과와 results/week3의 보고용 실행 결과를 정리).
week2는 학습을 다시 돌리지 않는다. 학습은 python src/week2_train.py --epochs 60 --batch 4 --name step3 (GPU 서버에서 실행)."""
import argparse, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["week1", "week2", "week3"])
ap.add_argument("--annotation"); ap.add_argument("--tool", choices=["CVAT", "Roboflow"]); ap.add_argument("--relabel")
a = ap.parse_args()
STEPS = {
    "week1": [["src/week1_audit.py"], ["src/week1_neardup.py"], ["src/week1_figures.py"], ["report/build_week1_report.py", "--pdf"]],
    "week3": [["src/week3_eval.py"], ["src/week3_compare.py"], ["report/build_week3_report.py", "--pdf"], ["src/week3_package.py"]],
    "week2": [["src/week2_prepare.py"], ["src/week2_checks.py", "--revalidate"], ["src/week2_curve.py"], ["report/build_week2_report.py", "--pdf"]],
}
steps = STEPS[a.step]
if a.step == "week2" and a.annotation:
    steps = steps[:3] + [["src/week2_annotation.py", "import", "--export", a.annotation, "--tool", a.tool or ap.error("--tool이 필요합니다")] + (["--relabel", a.relabel] if a.relabel else [])] + steps[3:]
for s in steps:
    print(">>", " ".join(s))
    subprocess.run([sys.executable] + s, check=True)
if a.step == "week2" and os.path.exists("results/week2/annotation/annotation_summary.json"):
    subprocess.run([sys.executable, "src/week2_package.py"], check=True)
