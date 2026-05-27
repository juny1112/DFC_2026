#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# ============================================================
# 입력 경로
# ============================================================
MEDIAN_ABUSE_XLSX = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95\Median_Abuse_user.xlsx"

T95_CSV = (
    r"G:\공유 드라이브\BSG_DFC_result\combined"
    r"\DFC_완충후이동주차\t95\t95_before_after_delta_combined.csv"
)

# ============================================================
# 출력 경로: summary CSV 하나만 저장
# ============================================================
OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95"
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "Median_Abuse_user_t95_summary.csv")


# ============================================================
# 파일명 정규화
# - 확장자 제거
# - _CR, _R, _r, _DFC 등이 붙어 있어도 같은 base_key로 매칭
# ============================================================
def normalize_file_key(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]

    for suf in ["_DFC", "_dfc", "_CR", "_cr", "_R", "_r"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ========================================================
    # 1. Median_Abuse_user.xlsx 읽기
    # ========================================================
    med = pd.read_excel(MEDIAN_ABUSE_XLSX)

    # 실제 파일 기준 컬럼명:
    # Unnamed: 0    -> Median 1, Median 2, Abuse 1, Abuse 2 ...
    # synthetic_vid -> 대표 synthetic user ID
    # base_vid      -> 실제 파일명 base key
    med = med.rename(columns={
        "Unnamed: 0": "user_group",
        "synthetic_vid": "synthetic_vid",
        "base_vid": "file"
    })

    required_med_cols = ["user_group", "synthetic_vid", "file"]
    missing = [c for c in required_med_cols if c not in med.columns]
    if missing:
        raise ValueError(f"Median_Abuse_user.xlsx 필수 컬럼 누락: {missing}")

    # user_group, synthetic_vid는 첫 행만 있고 아래가 비어있을 수 있으므로 채움
    med["user_group"] = med["user_group"].ffill()
    med["synthetic_vid"] = med["synthetic_vid"].ffill()

    # file 없는 행 제거
    med = med.dropna(subset=["file"]).copy()

    # 매칭용 key 생성
    med["file_key"] = med["file"].apply(normalize_file_key)

    # ========================================================
    # 2. t95_before_after_delta_combined.csv 읽기
    # ========================================================
    t95 = pd.read_csv(T95_CSV)

    required_t95_cols = ["file", "t95_before_h", "t95_after_h", "delta_t_h"]
    missing = [c for c in required_t95_cols if c not in t95.columns]
    if missing:
        raise ValueError(f"t95 CSV 필수 컬럼 누락: {missing}")

    t95 = t95.copy()
    t95["file_key"] = t95["file"].apply(normalize_file_key)

    # 숫자형 변환
    for c in ["t95_before_h", "t95_after_h", "delta_t_h"]:
        t95[c] = pd.to_numeric(t95[c], errors="coerce")

    # 중복 file_key가 있으면 첫 번째 값만 사용
    dup_keys = t95["file_key"][t95["file_key"].duplicated()].unique()
    if len(dup_keys) > 0:
        print("[WARN] t95 CSV에 중복 file_key가 있습니다. 첫 번째 값만 사용합니다.")
        print(dup_keys[:20])

    t95_sub = (
        t95[["file_key", "t95_before_h", "t95_after_h", "delta_t_h"]]
        .drop_duplicates(subset=["file_key"], keep="first")
        .copy()
    )

    # ========================================================
    # 3. Median/Abuse 사용자 파일 목록과 t95 결과 매칭
    # ========================================================
    detail = med.merge(
        t95_sub,
        on="file_key",
        how="left"
    )

    # 보기 좋은 컬럼명
    detail = detail.rename(columns={
        "file": "base_vid",
        "t95_before_h": "before_h",
        "t95_after_h": "after_h",
        "delta_t_h": "delta_t_h"
    })

    # 매칭 실패 확인
    unmatched = detail[detail["before_h"].isna() | detail["after_h"].isna()]
    if len(unmatched) > 0:
        print("\n[WARN] 매칭 실패 파일이 있습니다.")
        print(
            unmatched[
                ["user_group", "synthetic_vid", "base_vid", "file_key"]
            ].to_string(index=False)
        )

    # ========================================================
    # 4. 사용자별 summary 계산
    # ========================================================
    summary = (
        detail
        .groupby(["user_group", "synthetic_vid"], dropna=False)
        .agg(
            n_files=("base_vid", "count"),
            n_matched=("before_h", lambda x: x.notna().sum()),
            total_before_h=("before_h", "sum"),
            total_after_h=("after_h", "sum"),
            total_delta_t_h=("delta_t_h", "sum"),
        )
        .reset_index()
    )

    # before 대비 after가 몇 % 줄었는지
    # reduction_percent = (before - after) / before * 100
    # 여기서 delta_t_h = before - after 라고 가정
    summary["reduction_percent"] = np.where(
        summary["total_before_h"] > 0,
        summary["total_delta_t_h"] / summary["total_before_h"] * 100,
        np.nan
    )

    # 반올림
    round_cols = [
        "total_before_h",
        "total_after_h",
        "total_delta_t_h",
        "reduction_percent"
    ]
    summary[round_cols] = summary[round_cols].round(3)

    # ========================================================
    # 5. summary CSV 하나만 저장
    # ========================================================
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n[SAVE]")
    print(f"summary CSV : {OUT_SUMMARY_CSV}")

    print("\n[SUMMARY]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()