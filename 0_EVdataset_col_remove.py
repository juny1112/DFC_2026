#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# 출력 폴더: 이전처럼 하나로 통일
# ─────────────────────────────────────────────
OUTPUT_DIR = Path(r"Z:\SamsungSTF\Processed_Data\DFC\Data_release")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Fig. A/B/C base key
# ─────────────────────────────────────────────
TARGET_BASE_KEYS = [
    "bms_01241228090_2023-04",           # Fig. A
    "bms_01241248817_2023-04",           # Fig. B
    "bms_altitude_01241248932_2024-05",  # Fig. C
]

# ─────────────────────────────────────────────
# nonDFC 입력 폴더
# 실제로는 CR_parsing 파일에서 charging/rest만 제거해서 nonDFC로 저장
# ─────────────────────────────────────────────
CR_INPUT_DIRS = [
    Path(r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing"),
    Path(r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing"),
]

# ─────────────────────────────────────────────
# DFC 적용본 입력 폴더
# ─────────────────────────────────────────────
DFC_INPUT_DIRS = [
    Path(r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차"),
    Path(r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차"),
]

# ─────────────────────────────────────────────
# nonDFC로 저장할 컬럼
# - CR 파일에서 charging/rest는 제거
# - pack_volt 포함
# - speed 포함
# ─────────────────────────────────────────────
KEEP_COLS_NONDFC = [
    "time",
    "pack_current",
    "pack_volt",
    "speed",
    "soc",
    "mod_temp_list",
    "int_temp",
    "ext_temp",
]

# ─────────────────────────────────────────────
# DFC 적용본에서 저장할 컬럼
# - pack_volt 포함
# - speed 포함
# ─────────────────────────────────────────────
KEEP_COLS_DFC = [
    "time",
    "pack_current",
    "pack_volt",
    "speed",
    "soc",
    "mod_temp_list",
    "int_temp",
    "ext_temp",
]


def find_file(input_dirs, candidate_names):
    """
    여러 입력 폴더에서 후보 파일명들을 순서대로 찾음.
    찾으면 Path 반환, 못 찾으면 None.
    """
    for input_dir in input_dirs:
        for name in candidate_names:
            path = input_dir / name
            if path.exists():
                return path
    return None


def reduce_csv(in_path: Path, out_path: Path, keep_cols: list[str]):
    """
    CSV에서 keep_cols 중 실제 존재하는 컬럼만 남겨 저장.
    """
    header = pd.read_csv(in_path, nrows=0)
    existing_cols = list(header.columns)

    usecols = [c for c in keep_cols if c in existing_cols]
    missing_cols = [c for c in keep_cols if c not in existing_cols]

    if not usecols:
        print(f"[SKIP] No keep columns found: {in_path}")
        return

    df = pd.read_csv(in_path, usecols=usecols)
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {out_path}")
    print(f"       source      : {in_path}")
    print(f"       kept columns: {list(df.columns)}")
    if missing_cols:
        print(f"       missing cols : {missing_cols}")


def process_non_dfc_from_cr(base_key: str):
    """
    CR_parsing 파일에서 charging/rest를 제외한 nonDFC 파일 저장.
    입력:  *_CR.csv
    출력:  *_nonDFC.csv
    """
    candidate_names = [
        f"{base_key}_CR.csv",
    ]

    in_path = find_file(CR_INPUT_DIRS, candidate_names)

    if in_path is None:
        print(f"[MISSING nonDFC/CR] {base_key}_CR.csv")
        return

    out_path = OUTPUT_DIR / f"{base_key}_nonDFC.csv"
    reduce_csv(in_path, out_path, KEEP_COLS_NONDFC)


def process_dfc(base_key: str):
    """
    DFC 적용본 저장.
    입력:  *_DFC.csv
    출력:  *_DFC.csv
    """
    candidate_names = [
        f"{base_key}_DFC.csv",
    ]

    in_path = find_file(DFC_INPUT_DIRS, candidate_names)

    if in_path is None:
        print(f"[MISSING DFC] {base_key}_DFC.csv")
        return

    out_path = OUTPUT_DIR / f"{base_key}_DFC.csv"
    reduce_csv(in_path, out_path, KEEP_COLS_DFC)


def main():
    for base_key in TARGET_BASE_KEYS:
        print(f"\n========== {base_key} ==========")

        process_non_dfc_from_cr(base_key)
        process_dfc(base_key)

    print("\n[DONE] Saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()