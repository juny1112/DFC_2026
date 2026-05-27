#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Input / output
# ─────────────────────────────────────────────────────────────
INPUT_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95\t95_before_after_delta_combined.csv"

OUT_DIR = os.path.dirname(INPUT_CSV)

# 1) Non-DFC t_FC 기준으로 95%, 90% 파일을 선택한 표
OUT_BEFORE_WIDE_CSV = os.path.join(OUT_DIR, "Suppl_Table_tFC.csv")

# 2) Δt_FC 기준으로 95%, 90% 파일을 선택한 표
OUT_DELTA_WIDE_CSV = os.path.join(OUT_DIR, "Suppl_Table_delta_tFC.csv")

# 선택 파일 및 정확한 값 확인용 상세 파일(optional)

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────
COL_FILE    = "file"
COL_BEFORE  = "t95_before_h"   # Non-DFC t_FC
COL_AFTER   = "t95_after_h"    # DFC t_FC
COL_DELTA   = "delta_t_h"      # Δt_FC
COL_CLUSTER = "monthly_cluster"

# Cluster definition and display order
GROUPS = [
    ("All months",       "All", None),
    ("Long R_FC",        2,     2),
    ("Frequent R_FC",    1,     1),
    ("Minimal R_FC",     0,     0),
]

# Wide table columns inside each group
METRICS_WITH_FILE = [
    ("Non-DFC t_FC (hours)", "Non-DFC_tFC_h"),
    ("DFC t_FC (hours)",     "DFC_tFC_h"),
    ("Δt_FC (hours)",        "Delta_tFC_h"),
    ("selected_file",        "selected_file"),
]


def select_row_by_percentile(df_group: pd.DataFrame, q: float, basis_col: str) -> tuple[float, pd.Series]:
    """
    q percentile is calculated from basis_col within each group.
    Because percentile values are often interpolated and may not correspond
    to an actual file, the actual file closest to that percentile value is selected.

    Tie breaker:
      1) smallest absolute distance from percentile
      2) larger basis_col value
      3) file name alphabetically
    """
    q_value = df_group[basis_col].quantile(q)

    tmp = df_group.copy()
    tmp["_absdiff_from_q"] = (tmp[basis_col] - q_value).abs()

    selected = tmp.sort_values(
        by=["_absdiff_from_q", basis_col, COL_FILE],
        ascending=[True, False, True],
    ).iloc[0]

    return q_value, selected


def build_detail_table(df: pd.DataFrame, basis_col: str, basis_name: str) -> pd.DataFrame:
    records = []

    for group_name, cluster_label, cluster_value in GROUPS:
        if cluster_value is None:
            sub = df.copy()
        else:
            sub = df[df[COL_CLUSTER] == cluster_value].copy()

        if sub.empty:
            print(f"[WARN] No rows for {group_name}. Skipped.")
            continue

        # 95th and 90th percentile rows:
        # select one actual file by basis_col, then take before/after/delta from that same file.
        for stat_label, q in [("95%", 0.95), ("90%", 0.90)]:
            q_value, row = select_row_by_percentile(sub, q, basis_col)

            records.append({
                "group": group_name,
                "monthly_cluster": cluster_label,
                "statistic": stat_label,
                "selection_basis": basis_name,
                "calculation_basis": f"file closest to {int(q * 100)}th percentile of {basis_col} within group",
                "n_files": len(sub),
                "selected_file": row[COL_FILE],
                "quantile_value": q_value,
                "Non-DFC_tFC_h": row[COL_BEFORE],
                "DFC_tFC_h": row[COL_AFTER],
                "Delta_tFC_h": row[COL_DELTA],
                "check_before_minus_after": row[COL_BEFORE] - row[COL_AFTER],
            })

        # Mean rows: keep previous method, i.e., column-wise means.
        records.append({
            "group": group_name,
            "monthly_cluster": cluster_label,
            "statistic": "Mean",
            "selection_basis": basis_name,
            "calculation_basis": "column-wise mean within group",
            "n_files": len(sub),
            "selected_file": "",
            "quantile_value": np.nan,
            "Non-DFC_tFC_h": sub[COL_BEFORE].mean(),
            "DFC_tFC_h": sub[COL_AFTER].mean(),
            "Delta_tFC_h": sub[COL_DELTA].mean(),
            "check_before_minus_after": sub[COL_BEFORE].mean() - sub[COL_AFTER].mean(),
        })

    out = pd.DataFrame(records)

    numeric_cols = [
        "quantile_value",
        "Non-DFC_tFC_h",
        "DFC_tFC_h",
        "Delta_tFC_h",
        "check_before_minus_after",
    ]
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def _format_value_for_wide(value, round_digits: int = 0):
    if pd.isna(value):
        return ""
    if round_digits == 0:
        return int(round(float(value)))
    return round(float(value), round_digits)


def build_wide_table(detail_df: pd.DataFrame, round_digits: int = 0) -> pd.DataFrame:
    """
    Create a table-shaped CSV like the manuscript table layout.

    CSV cannot contain merged cells, so the first row places each group label
    at the first column of its 4-column block and leaves the remaining cells blank.
    The fourth column in each group block records the selected file.
    """
    header_group = [""]
    header_metric = [""]

    for group_name, _, _ in GROUPS:
        header_group += [group_name, "", "", ""]
        header_metric += [label for label, _ in METRICS_WITH_FILE]

    rows = [header_group, header_metric]

    for stat in ["95%", "90%", "Mean"]:
        row_out = [stat]
        for group_name, _, _ in GROUPS:
            sub = detail_df[(detail_df["group"] == group_name) & (detail_df["statistic"] == stat)]
            if sub.empty:
                row_out += [""] * len(METRICS_WITH_FILE)
                continue

            rec = sub.iloc[0]

            # Display values are rounded for manuscript-style readability.
            # To make the printed table satisfy Non-DFC - DFC = Delta exactly,
            # Delta is calculated from the displayed Non-DFC and DFC values.
            before_disp = _format_value_for_wide(rec["Non-DFC_tFC_h"], round_digits=round_digits)
            after_disp  = _format_value_for_wide(rec["DFC_tFC_h"],     round_digits=round_digits)
            if before_disp == "" or after_disp == "":
                delta_disp = ""
            else:
                delta_disp = before_disp - after_disp

            row_out += [
                before_disp,
                after_disp,
                delta_disp,
                rec["selected_file"] if isinstance(rec["selected_file"], str) else "",
            ]
        rows.append(row_out)

    return pd.DataFrame(rows)


def load_input_csv(input_csv: str) -> pd.DataFrame:
    # usecols prevents old pasted-table columns from affecting the calculation.
    required = [COL_FILE, COL_BEFORE, COL_AFTER, COL_DELTA, COL_CLUSTER]
    df = pd.read_csv(input_csv, usecols=required)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Numeric conversion
    for c in [COL_BEFORE, COL_AFTER, COL_DELTA, COL_CLUSTER]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep valid rows only
    df = df.dropna(subset=[COL_FILE, COL_BEFORE, COL_AFTER, COL_DELTA]).copy()
    return df


def main():
    df = load_input_csv(INPUT_CSV)

    # Table 1: select percentile files by Non-DFC t_FC
    before_detail = build_detail_table(
        df,
        basis_col=COL_BEFORE,
        basis_name="Non-DFC t_FC percentile basis",
    )
    before_wide = build_wide_table(before_detail, round_digits=0)

    # Table 2: select percentile files by Δt_FC
    delta_detail = build_detail_table(
        df,
        basis_col=COL_DELTA,
        basis_name="Delta t_FC percentile basis",
    )
    delta_wide = build_wide_table(delta_detail, round_digits=0)

    before_wide.to_csv(OUT_BEFORE_WIDE_CSV, index=False, header=False, encoding="utf-8-sig")
    delta_wide.to_csv(OUT_DELTA_WIDE_CSV, index=False, header=False, encoding="utf-8-sig")

    print(f"[SAVE] before-basis wide table -> {OUT_BEFORE_WIDE_CSV}")
    print(f"[SAVE] delta-basis wide table  -> {OUT_DELTA_WIDE_CSV}")

    print("\n[BEFORE-BASIS WIDE PREVIEW]")
    print(before_wide.to_string(index=False, header=False))
    print("\n[DELTA-BASIS WIDE PREVIEW]")
    print(delta_wide.to_string(index=False, header=False))


if __name__ == "__main__":
    main()
