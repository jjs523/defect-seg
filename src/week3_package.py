"""3주차 최종 제출 zip: 리포트 PDF(기준서 수정본은 부록), 제출 가중치(clean_s1), 코드와 결과 파일."""
import glob, os, zipfile
PDFS = ["report/Step3_최종리포트_22114553_전진성.pdf"]  # 기준서 수정본은 리포트 부록에 포함
OUT = "dist/Step3_3주차_최종제출_22114553_전진성.zip"
for p in PDFS:
    assert os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime("results/week3/week3_summary.json"), f"{p}가 없거나 결과보다 오래되었습니다. python main.py week3"
items = [(p, os.path.basename(p)) for p in PDFS] + [("results/improve/train_s1/weights/best.pt", "best.pt")]
code = ["main.py", "setup.sh", "requirements.lock.txt", "dataset/data.yaml", "src/common.py", "src/week2_prepare.py", "src/week2_train.py",
        "src/improve_train.sh", "src/improve_eval.py", "src/week3_eval.py", "src/week3_package.py", "report/build_week3_report.py",
        "annotation/label_fixes.csv", "results/week1/hold_list.csv", "results/week1/class_examples.png", "results/week2/integrity.json",
        "results/week2/split_checks.json", "results/week2/annotation/annotation_summary.json"]
res = [p for p in sorted(glob.glob("results/week3/**/*", recursive=True) + glob.glob("results/improve/*") + glob.glob("results/improve/train_s1/*"))
       if os.path.isfile(p) and not p.endswith(".pt") and "/val_runs/" not in p]
res += [p for p in ("results/paper/test_metrics.json", "results/paper/thresholds.json") if os.path.exists(p)]
items += [(p, f"code_results/{p}") for p in code + res]
os.makedirs("dist", exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in items:
        z.write(src, arc)
print("ZIP", OUT, os.path.getsize(OUT) // 1024, "KB", len(items), "files")
