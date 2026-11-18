"""파일명 규칙과 경로를 한곳에 둔다."""
import re

NAMES = ["crack", "efflorescence", "spall_detach"]
SEED = 42


def origin_id(stem):
    """파일명에서 원본 프레임 ID를 뽑는다. 예: URT007_2022_01_13_001_5468_efflo_hflip -> URT007_2022_01_13_001_5468"""
    m = re.match(r"^(U[A-Z]{2}\d+_\d{4}_\d{2}_\d{2}_\d+_\d+)", stem)
    return m.group(1) if m else stem
