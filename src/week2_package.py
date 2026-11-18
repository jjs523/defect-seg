"""2주차 제출 zip을 만든다. 직접 주석 결과와 최종 PDF가 없으면 만들지 않는다."""
import glob, os, sys, zipfile
PDF = "report/Step3_2주차과제_보고서_22114553_전진성.pdf"
OUT = "dist/Step3_2주차과제_제출_22114553_전진성.zip"
if not os.path.exists("results/week2/annotation/annotation_summary.json") or not os.path.exists(PDF):
    sys.exit("직접 주석 결과 또는 최종 PDF가 없어 제출 zip을 만들지 않았습니다.")
if os.path.getmtime(PDF) < os.path.getmtime("results/week2/annotation/annotation_summary.json"):
    sys.exit("PDF가 주석 결과보다 오래되었습니다. python main.py week2로 다시 만드세요.")
items = [(PDF, os.path.basename(PDF)), ("results/week2/train/weights/best.pt", "best.pt")]
code = ["main.py", "setup.sh", "requirements.lock.txt", "src/common.py", "report/build_week2_report.py", "annotation/batch_list.csv", "annotation/label_fixes.csv", "results/week1/hold_list.csv", "dataset/data.yaml"] + sorted(glob.glob("src/week2_*.py"))
res = [p for p in sorted(glob.glob("results/week2/**/*", recursive=True)) if os.path.isfile(p) and not p.endswith(".pt")]
items += [(p, f"code_results/{p}") for p in code + res]
os.makedirs("dist", exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in items:
        z.write(src, arc)
print("ZIP", OUT, os.path.getsize(OUT) // 1024, "KB", len(items), "files")
