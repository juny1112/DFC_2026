#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified DFC pipeline
====================
Input  : raw/non-DFC BMS CSV file(s)
Output : DFC-synthesized CSV file(s) + optional event summary CSV

Pipeline:
  raw non-DFC CSV
    -> type normalization
    -> transient SOC=0 correction
    -> invalid-file screening
    -> charging/rest parsing
    -> event labeling: R_charg, R_partial_charg, R_FC, R_other_rest
    -> DFC synthesis
    -> *_DFC.csv + dfc_features_summary.csv

Label convention:
  R_charg         : full-charge segment
  R_partial_charg : partial-charge segment
  R_FC            : post-full-charge rest / full-charge dwell segment
  R_other_rest    : rest segment not assigned to full-charge or partial-charge context

Notes:
  - The old R_aftercharg label is renamed to R_FC.
  - The old R_uncharg label is renamed to R_other_rest.
  - By default, intermediate label columns are removed from final DFC outputs.
  - Set keep_labels=True if you want to inspect intermediate labels in output files.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

TIME_FMT = "%Y-%m-%d %H:%M:%S"
NUM_COLS = ["soc", "pack_current", "pack_volt", "speed"]

LABEL_FULL_CHARGE = "R_charg"
LABEL_PARTIAL_CHARGE = "R_partial_charg"
LABEL_FC = "R_FC"
LABEL_OTHER_REST = "R_other_rest"

SUMMARY_COLUMNS = [
    "file_stem",
    "id_token",
    "ym",
    "status",
    "message",
    "delta_t95_event_N",
    "delta_t95_event_mean_h",
    "delta_t95_event_std_h",
    "delta_t95_event_sum_h",
]


@dataclass(frozen=True)
class DFCConfig:
    full_soc_threshold: float = 95.0
    delay_soc_threshold: float = 80.0
    departure_buffer_h: float = 1.0
    fullcharge_parking_mode: bool = True
    merge_gap_minutes: float = 30.0
    min_other_rest_hours_for_merge: Optional[float] = 7.0
    heater_gap_minutes: float = 5.0
    heater_max_duration_minutes: float = 30.0
    remove_label_columns: bool = True


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def _as_int_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def get_blocks(series: pd.Series | np.ndarray) -> list[tuple[int, int]]:
    """Return consecutive nonzero/True blocks as inclusive (start, end) pairs."""
    arr = np.asarray(series)
    if arr.dtype != bool:
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0).to_numpy() != 0

    blocks: list[tuple[int, int]] = []
    in_block = False
    start = 0
    for i, flag in enumerate(arr):
        if flag and not in_block:
            start = i
            in_block = True
        elif (not flag) and in_block:
            blocks.append((start, i - 1))
            in_block = False
    if in_block:
        blocks.append((start, len(arr) - 1))
    return blocks


def parse_id_token_and_ym(path: str | Path) -> tuple[str, str]:
    """
    Parse id_token and YYYY-MM from filenames such as:
      bms_01241228021_2023-02.csv
      bms_01241228021_2023-02_DFC.csv
      bms_altitude_01241248932_2024-05.csv
    """
    stem = Path(path).stem
    m = re.search(r"bms_(?:altitude_)?(?P<id>\d+)_(?P<ym>\d{4}-\d{2})", stem, re.I)
    if not m:
        return "unknown", "0000-00"
    return m.group("id"), m.group("ym")


def output_name_for_input(path: str | Path, suffix: str = "_DFC") -> str:
    stem = Path(path).stem
    stem = re.sub(r"(_CR|_cr|_r|_R|_DFC|_dfc)$", "", stem)
    return f"{stem}{suffix}.csv"


# -----------------------------------------------------------------------------
# Step 1. Raw parsing: type normalization, SOC=0 correction, charging/rest parsing
# -----------------------------------------------------------------------------
def prep_types_once(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "time" not in data.columns:
        raise ValueError("missing required column: time")
    data["time"] = pd.to_datetime(data["time"].astype(str).str.strip(), format=TIME_FMT, errors="raise")
    for col in NUM_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def fix_soc_zero(data: pd.DataFrame) -> pd.DataFrame:
    """Correct short transient SOC=0 dropout segments using linear interpolation."""
    data = data.copy()
    if "soc" not in data.columns:
        raise ValueError("missing required column: soc")

    i = 0
    n = len(data)
    while i < n - 1:
        if data.loc[i, "soc"] > 0.5 and data.loc[i + 1, "soc"] == 0:
            start = i + 1
            end = start
            while end < n and data.loc[end, "soc"] == 0:
                end += 1

            if "pack_current" in data.columns:
                data.loc[start:end - 1, "pack_current"] = 0

            if end < n:
                data.loc[start:end, "soc"] = np.linspace(
                    data.loc[i, "soc"], data.loc[end, "soc"], end - i + 1
                )[1:]
                if "pack_volt" in data.columns:
                    data.loc[start:end, "pack_volt"] = np.linspace(
                        data.loc[i, "pack_volt"], data.loc[end, "pack_volt"], end - i + 1
                    )[1:]
        i += 1
    return data


def is_invalid_data(data: pd.DataFrame) -> bool:
    dsoc = data["soc"].diff().abs()
    dtime = data["time"].diff()
    return bool(((dsoc >= 10) & (dtime >= pd.Timedelta(hours=12))).any())


def parse_charging(data: pd.DataFrame) -> pd.DataFrame:
    """Parse charging using pack_current < 0 and speed == 0, with short-gap cleanup."""
    data = data.copy()
    required = {"pack_current", "speed", "time"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns for charging parsing: {missing}")

    n = len(data)
    if n == 0:
        data["charging"] = np.int8(0)
        return data

    charging = ((data["pack_current"] < 0) & (data["speed"] == 0)).astype(np.int8).to_numpy()
    data["charging"] = charging
    t = data["time"]

    # Fill non-charging gaps <= 1 min between charging blocks.
    grp = np.cumsum(np.r_[True, charging[1:] != charging[:-1]])
    data["_chg_grp"] = grp
    g = data.groupby("_chg_grp", sort=False)
    g_val = g["charging"].first()
    g_first = g.apply(lambda x: x.index[0])
    g_last = g.apply(lambda x: x.index[-1])
    g_t_start = t.loc[g_first.values].to_numpy()
    g_t_end = t.loc[g_last.values].to_numpy()

    zero_groups = g_val[g_val == 0].index.to_numpy()
    if len(zero_groups) > 0:
        zero_dur = g_t_end[zero_groups - 1] - g_t_start[zero_groups - 1]
        for gid in zero_groups[zero_dur <= np.timedelta64(1, "m")]:
            data.loc[int(g_first.loc[gid]):int(g_last.loc[gid]), "charging"] = 1

    # Remove charging blocks <= 5 min.
    charging2 = data["charging"].to_numpy().astype(np.int8)
    grp2 = np.cumsum(np.r_[True, charging2[1:] != charging2[:-1]])
    data["_chg_grp2"] = grp2
    g2 = data.groupby("_chg_grp2", sort=False)
    g2_val = g2["charging"].first()
    g2_first = g2.apply(lambda x: x.index[0])
    g2_last = g2.apply(lambda x: x.index[-1])
    g2_t_start = t.loc[g2_first.values].to_numpy()
    g2_t_end = t.loc[g2_last.values].to_numpy()

    one_groups = g2_val[g2_val == 1].index.to_numpy()
    if len(one_groups) > 0:
        one_dur = g2_t_end[one_groups - 1] - g2_t_start[one_groups - 1]
        for gid in one_groups[one_dur <= np.timedelta64(5, "m")]:
            data.loc[int(g2_first.loc[gid]):int(g2_last.loc[gid]), "charging"] = 0

    data.drop(columns=["_chg_grp", "_chg_grp2"], inplace=True, errors="ignore")
    data["charging"] = data["charging"].astype(np.int8)
    return data


def parse_rest(data: pd.DataFrame) -> pd.DataFrame:
    """Parse rest segments from charging, time gaps, and near-zero current segments."""
    data = data.copy()
    required = {"charging", "pack_current", "time"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns for rest parsing: {missing}")

    n = len(data)
    if n == 0:
        data["rest"] = np.int8(0)
        return data

    data["rest"] = (data["charging"] == 1).astype(np.int8)
    t = data["time"]

    # Mark both sides of a >5 min sampling gap as rest.
    gap_idx = np.flatnonzero((t.diff() > pd.Timedelta(minutes=5)).to_numpy())
    if gap_idx.size > 0:
        mark = np.unique(np.r_[gap_idx - 1, gap_idx])
        mark = mark[(mark >= 0) & (mark < n)]
        data.loc[mark, "rest"] = 1

    # Mark 0 <= pack_current <= 1 A blocks lasting >=10 min as rest.
    abnormal = ((data["pack_current"] >= 0) & (data["pack_current"] <= 1)).to_numpy()
    if abnormal.any():
        grp = np.cumsum(np.r_[True, abnormal[1:] != abnormal[:-1]])
        data["_ab_grp"] = grp
        g = data.groupby("_ab_grp", sort=False)
        g_val = g.apply(lambda x: bool(0 <= x["pack_current"].iloc[0] <= 1))
        g_first = g.apply(lambda x: x.index[0])
        g_last = g.apply(lambda x: x.index[-1])
        for gid in g_val[g_val].index:
            s, e = int(g_first.loc[gid]), int(g_last.loc[gid])
            if (t.iloc[e] - t.iloc[s]) >= pd.Timedelta(minutes=10):
                data.loc[s:e, "rest"] = 1
        data.drop(columns=["_ab_grp"], inplace=True, errors="ignore")

    # Fill rest gaps <= 1 min between rest blocks.
    rest = data["rest"].to_numpy().astype(np.int8)
    grp = np.cumsum(np.r_[True, rest[1:] != rest[:-1]])
    data["_r_grp"] = grp
    g = data.groupby("_r_grp", sort=False)
    g_val = g["rest"].first()
    g_first = g.apply(lambda x: x.index[0])
    g_last = g.apply(lambda x: x.index[-1])

    for gid in g_val[g_val == 0].index.to_numpy():
        prev_gid, next_gid = gid - 1, gid + 1
        if prev_gid not in g_val.index or next_gid not in g_val.index:
            continue
        if g_val.loc[prev_gid] != 1 or g_val.loc[next_gid] != 1:
            continue
        left_idx = int(g_last.loc[prev_gid])
        right_idx = int(g_first.loc[next_gid])
        if (t.iloc[right_idx] - t.iloc[left_idx]) <= pd.Timedelta(minutes=1):
            data.loc[int(g_first.loc[gid]):int(g_last.loc[gid]), "rest"] = 1

    data.drop(columns=["_r_grp"], inplace=True, errors="ignore")
    data["rest"] = data["rest"].astype(np.int8)
    return data


def parse_raw_non_dfc(data: pd.DataFrame) -> pd.DataFrame:
    data = prep_types_once(data)
    data = fix_soc_zero(data)
    if is_invalid_data(data):
        raise ValueError("invalid file: SOC change >=10 percentage points across a >=12 h gap")
    data = parse_charging(data)
    data = parse_rest(data)
    return data


# -----------------------------------------------------------------------------
# Step 2. Event labeling with renamed labels
# -----------------------------------------------------------------------------
def label_full_and_partial_charge(data: pd.DataFrame, config: DFCConfig) -> tuple[pd.DataFrame, list[int]]:
    data = data.copy()
    charging = _as_int_flag(data["charging"])
    charge_blocks = get_blocks(charging.to_numpy())

    full_indices: list[int] = []
    data[LABEL_FULL_CHARGE] = 0
    data[LABEL_PARTIAL_CHARGE] = 0

    for start, end in charge_blocks:
        # Keep original intent: inspect a small neighborhood after charge end.
        lo = max(0, end - 3)
        hi = min(len(data) - 1, end + 6)
        is_full = (pd.to_numeric(data.loc[lo:hi, "soc"], errors="coerce") >= config.full_soc_threshold).any()
        if is_full:
            data.loc[start:end, LABEL_FULL_CHARGE] = 1
            full_indices.extend([start, end + 1])
        else:
            data.loc[start:end, LABEL_PARTIAL_CHARGE] = 1

    return data, sorted(set(full_indices))


def label_R_FC(data: pd.DataFrame, full: list[int], config: DFCConfig) -> pd.DataFrame:
    """Label post-full-charge dwell/rest segments as R_FC."""
    data = data.copy()
    data[LABEL_FC] = 0
    data["time"] = pd.to_datetime(data["time"], format=TIME_FMT, errors="coerce")

    n = len(data)

    # 1) Rest immediately following full-charge segment -> R_FC.
    for i in range(n - 1):
        if (
            data.loc[i, LABEL_FULL_CHARGE] == 1
            and data.loc[i + 1, LABEL_FULL_CHARGE] == 0
            and data.loc[i + 1, "rest"] == 1
        ):
            j = i
            while j < n and data.loc[j, "rest"] == 1:
                data.at[j, LABEL_FC] = 1
                j += 1

    # 2) Short pre-departure thermal-conditioning-like segment parsed as charging.
    limit_time = pd.Timedelta(minutes=config.heater_gap_minutes)
    limit_duration = pd.Timedelta(minutes=config.heater_max_duration_minutes)
    has_current = "pack_current" in data.columns

    for i in range(n - 1):
        if not (
            data.loc[i, LABEL_FULL_CHARGE] == 1
            and data.loc[i + 1, LABEL_FULL_CHARGE] == 1
            and data.loc[i + 1, "time"] - data.loc[i, "time"] > limit_time
        ):
            continue

        j = i
        while j < n and data.loc[j, LABEL_FULL_CHARGE] == 1:
            j += 1
        if j <= i + 1:
            continue

        duration = data.loc[j - 1, "time"] - data.loc[i + 1, "time"]
        soc_start = pd.to_numeric(data.loc[i + 1, "soc"], errors="coerce")
        if pd.isna(soc_start) or soc_start < config.full_soc_threshold:
            continue

        if has_current:
            pc = pd.to_numeric(data.loc[i + 1:j - 1, "pack_current"], errors="coerce")
            if (pc <= 0).all():
                continue

        if duration <= limit_duration:
            data.loc[i:j - 1, LABEL_FC] = 1
            k = i + 1
            while k < n and data.loc[k, LABEL_FULL_CHARGE] == 1:
                data.at[k, LABEL_FULL_CHARGE] = 0
                data.at[k, LABEL_FC] = 1
                k += 1

    return data


def label_other_rest_and_merge(data: pd.DataFrame, config: DFCConfig) -> pd.DataFrame:
    """Label non-full-charge rest as R_other_rest, optionally merge move-and-park behavior into R_FC."""
    data = data.copy()
    data[LABEL_OTHER_REST] = 0

    rest_mask = (
        (_as_int_flag(data["rest"]) == 1)
        & (_as_int_flag(data[LABEL_FULL_CHARGE]) == 0)
        & (_as_int_flag(data[LABEL_PARTIAL_CHARGE]) == 0)
        & (_as_int_flag(data[LABEL_FC]) == 0)
    )
    data.loc[rest_mask, LABEL_OTHER_REST] = 1

    if not config.fullcharge_parking_mode:
        return data

    data["time"] = pd.to_datetime(data["time"], format=TIME_FMT, errors="coerce")
    time_limit = pd.Timedelta(minutes=config.merge_gap_minutes)
    min_rest_td = None
    if config.min_other_rest_hours_for_merge is not None and config.min_other_rest_hours_for_merge > 0:
        min_rest_td = pd.Timedelta(hours=config.min_other_rest_hours_for_merge)

    fc_blocks = get_blocks(data[LABEL_FC].fillna(0).astype(int).to_numpy())
    other_blocks = get_blocks(data[LABEL_OTHER_REST].fillna(0).astype(int).to_numpy())
    full_blocks = get_blocks(data[LABEL_FULL_CHARGE].fillna(0).astype(int).to_numpy())

    def passes_duration(start: int, end: int) -> bool:
        if min_rest_td is None:
            return True
        return (data.loc[end, "time"] - data.loc[start, "time"]) >= min_rest_td

    # Merge first R_other_rest block after R_FC when gap is small and duration condition passes.
    for _, fc_end in fc_blocks:
        next_other = next(((s, e) for s, e in other_blocks if s > fc_end), None)
        if not next_other:
            continue
        s, e = next_other
        gap = data.loc[s, "time"] - data.loc[fc_end, "time"]
        if gap <= time_limit and passes_duration(s, e):
            data.loc[fc_end + 1:e, LABEL_FC] = 1
            data.loc[fc_end + 1:e, LABEL_OTHER_REST] = 0

    # Same rule anchored at the end of a full-charge block.
    for _, full_end in full_blocks:
        next_other = next(((s, e) for s, e in other_blocks if s > full_end), None)
        if not next_other:
            continue
        s, e = next_other
        gap = data.loc[s, "time"] - data.loc[full_end, "time"]
        if gap <= time_limit and passes_duration(s, e):
            data.loc[full_end:e, LABEL_FC] = 1
            data.loc[full_end:e, LABEL_OTHER_REST] = 0

    return data


def label_events(data: pd.DataFrame, config: DFCConfig) -> pd.DataFrame:
    data, full = label_full_and_partial_charge(data, config)
    data = label_R_FC(data, full, config)
    data = label_other_rest_and_merge(data, config)
    return data


# -----------------------------------------------------------------------------
# Step 3. DFC synthesis
# -----------------------------------------------------------------------------
def remove_redundant_R_FC_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce nonprotected long R_FC runs to first/last row before DFC time shifting.
    R_FC runs following a full-charge segment that started at SOC >=95 are preserved.
    """
    if LABEL_FC not in data.columns:
        return data

    data = data.copy()
    s = data[LABEL_FC].fillna(0).astype(int)
    if len(s) == 0:
        return data

    grp = (s != s.shift(fill_value=s.iloc[0])).cumsum()
    group_sizes = grp.map(grp.value_counts())
    pos_from_start = data.groupby(grp).cumcount()
    pos_from_end = data.iloc[::-1].groupby(grp.iloc[::-1]).cumcount()[::-1]

    protect_groups = set()
    if LABEL_FULL_CHARGE in data.columns and "soc" in data.columns:
        for g in grp[s == 1].unique():
            idxs = np.flatnonzero(grp.values == g)
            if len(idxs) == 0:
                continue
            start_idx = int(idxs[0])
            if start_idx == 0:
                continue
            prev_idx = start_idx - 1
            if int(data.loc[prev_idx, LABEL_FULL_CHARGE]) != 1:
                continue
            k = prev_idx
            while k > 0 and int(data.loc[k - 1, LABEL_FULL_CHARGE]) == 1:
                k -= 1
            soc_start = pd.to_numeric(data.loc[k, "soc"], errors="coerce")
            if pd.notna(soc_start) and soc_start >= 95:
                protect_groups.add(g)

    protect_flag = grp.isin(protect_groups)
    keep = (
        (s == 0)
        | ((s == 1) & protect_flag)
        | ((s == 1) & ~protect_flag & (group_sizes < 3))
        | ((s == 1) & ~protect_flag & (group_sizes >= 3) & ((pos_from_start == 0) | (pos_from_end == 0)))
    )
    return data.loc[keep].reset_index(drop=True)


def apply_dfc(data: pd.DataFrame, config: DFCConfig, collect_stats: bool = True) -> tuple[pd.DataFrame, list[dict], dict]:
    data = remove_redundant_R_FC_rows(data)
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"], format=TIME_FMT, errors="coerce")
    data["soc"] = pd.to_numeric(data["soc"], errors="coerce")

    full_arr = data[LABEL_FULL_CHARGE].fillna(0).astype(int).to_numpy()
    fc_arr = data[LABEL_FC].fillna(0).astype(int).to_numpy()
    soc_arr = data["soc"].to_numpy()
    n = len(data)

    full_blocks = get_blocks(full_arr)

    def any_fc_near(idx: int) -> bool:
        lo = max(0, idx - 1)
        hi = min(n - 1, idx + 1)
        return bool((fc_arr[lo:hi + 1] == 1).any())

    dfc_charge_blocks = [(s, e) for s, e in full_blocks if any_fc_near(e)]

    delay_pairs: list[tuple[int, int]] = []
    for start, end in dfc_charge_blocks:
        found = False
        for i in range(start, max(start, end)):
            if soc_arr[i] < config.delay_soc_threshold and soc_arr[i + 1] == config.delay_soc_threshold:
                delay_pairs.append((i + 1, end))
                found = True
                break
        if (not found) and soc_arr[start] >= config.delay_soc_threshold:
            delay_pairs.append((start, end))

    fc_blocks = get_blocks(fc_arr)
    fc_starts = np.asarray([s for s, _ in fc_blocks], dtype=int)
    fc_ends = np.asarray([e for _, e in fc_blocks], dtype=int)
    full_starts = np.asarray([s for s, _ in full_blocks], dtype=int)

    events: list[dict] = []
    margin = pd.Timedelta(hours=config.departure_buffer_h)

    for dstart, cend in delay_pairs:
        soc_d = pd.to_numeric(data.loc[dstart, "soc"], errors="coerce")
        if pd.notna(soc_d) and soc_d >= config.full_soc_threshold:
            continue

        # Next full-charge start after current charge end.
        pos_next_charge = np.searchsorted(full_starts, cend + 1, side="left")
        next_charge_start = full_starts[pos_next_charge] if pos_next_charge < len(full_starts) else None

        # First R_FC block starting at/after charge end.
        pos_fc = np.searchsorted(fc_starts, cend, side="left")
        if pos_fc >= len(fc_starts):
            continue
        fc_start = int(fc_starts[pos_fc])
        if next_charge_start is not None and fc_start >= next_charge_start:
            continue
        fc_end = int(fc_ends[pos_fc])

        t0 = data.loc[cend, "time"]
        t1 = data.loc[fc_end, "time"]
        if pd.isna(t0) or pd.isna(t1):
            continue

        delayed_time = t1 - t0 - margin
        if delayed_time <= pd.Timedelta(0) or dstart + 1 > cend:
            continue

        events.append({
            "delay_start_idx": int(dstart),
            "charge_end_idx": int(cend),
            "R_FC_start_idx": int(fc_start),
            "R_FC_end_idx": int(fc_end),
            "charge_end_time": t0,
            "R_FC_end_time": t1,
            "delay_hours": delayed_time.total_seconds() / 3600.0,
        })
        data.loc[dstart + 1:cend, "time"] = data.loc[dstart + 1:cend, "time"] + delayed_time

    delays = pd.to_numeric(pd.Series([e["delay_hours"] for e in events], dtype="float64"), errors="coerce").dropna()
    event_n = int(len(delays))
    stats = {
        "delta_t95_event_N": event_n,
        "delta_t95_event_mean_h": float(delays.mean()) if event_n > 0 else 0.0,
        "delta_t95_event_std_h": float(delays.std(ddof=1)) if event_n > 1 else 0.0,
        "delta_t95_event_sum_h": float(delays.sum()) if event_n > 0 else 0.0,
    }

    if config.remove_label_columns:
        data = data.drop(
            columns=[LABEL_FULL_CHARGE, LABEL_PARTIAL_CHARGE, LABEL_FC, LABEL_OTHER_REST],
            errors="ignore",
        )

    return data, events, stats


# -----------------------------------------------------------------------------
# Full one-file and folder APIs
# -----------------------------------------------------------------------------
def run_dfc_pipeline_on_dataframe(data: pd.DataFrame, config: DFCConfig) -> tuple[pd.DataFrame, list[dict], dict]:
    parsed = parse_raw_non_dfc(data)
    labeled = label_events(parsed, config)
    return apply_dfc(labeled, config, collect_stats=True)


def run_dfc_pipeline_on_file(input_path: str | Path, output_path: str | Path, config: DFCConfig) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    dfc_df, events, stats = run_dfc_pipeline_on_dataframe(raw, config)
    dfc_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    id_token, ym = parse_id_token_and_ym(input_path)
    row = {
        "file_stem": input_path.stem,
        "id_token": id_token,
        "ym": ym,
        "status": "ok",
        "message": "",
        **stats,
    }
    return row


def _worker(args: tuple[str, str, DFCConfig, bool]) -> dict:
    input_path, output_path, config, skip_existing = args
    if skip_existing and Path(output_path).exists():
        id_token, ym = parse_id_token_and_ym(input_path)
        return {
            "file_stem": Path(input_path).stem,
            "id_token": id_token,
            "ym": ym,
            "status": "skip_existing",
            "message": "output already exists",
            "delta_t95_event_N": 0,
            "delta_t95_event_mean_h": 0.0,
            "delta_t95_event_std_h": 0.0,
            "delta_t95_event_sum_h": 0.0,
        }
    try:
        return run_dfc_pipeline_on_file(input_path, output_path, config)
    except Exception as exc:
        id_token, ym = parse_id_token_and_ym(input_path)
        return {
            "file_stem": Path(input_path).stem,
            "id_token": id_token,
            "ym": ym,
            "status": "error",
            "message": str(exc),
            "delta_t95_event_N": 0,
            "delta_t95_event_mean_h": 0.0,
            "delta_t95_event_std_h": 0.0,
            "delta_t95_event_sum_h": 0.0,
        }


def collect_csv_files(input_path: str | Path, pattern: str = "*.csv") -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def process_path(
    input_path: str | Path,
    output_dir: str | Path,
    summary_csv_path: str | Path | None,
    config: DFCConfig,
    pattern: str = "*.csv",
    workers: int = 1,
    skip_existing: bool = True,
) -> pd.DataFrame:
    files = collect_csv_files(input_path, pattern=pattern)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in files:
        out_path = output_dir / output_name_for_input(p, suffix="_DFC")
        jobs.append((str(p), str(out_path), config, skip_existing))

    rows: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for job in tqdm(jobs, desc="DFC pipeline", unit="file"):
            rows.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="DFC pipeline (MP)", unit="file"):
                rows.append(fut.result())

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if summary_csv_path is None:
        summary_csv_path = output_dir / "dfc_features_summary.csv"
    summary_csv_path = Path(summary_csv_path)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    ok = int((summary["status"] == "ok").sum()) if not summary.empty else 0
    skipped = int((summary["status"] == "skip_existing").sum()) if not summary.empty else 0
    errors = int((summary["status"] == "error").sum()) if not summary.empty else 0
    print(f"[done] ok={ok}, skip_existing={skipped}, error={errors}")
    print(f"[save] summary -> {summary_csv_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified raw/non-DFC CSV -> DFC CSV pipeline")
    p.add_argument("--input", required=True, help="Input raw/non-DFC CSV file or folder")
    p.add_argument("--output-dir", required=True, help="Output folder for *_DFC.csv files")
    p.add_argument("--summary-csv", default=None, help="Summary CSV path. Default: <output-dir>/dfc_features_summary.csv")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern when input is a folder")
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing *_DFC.csv outputs")
    p.add_argument("--keep-labels", action="store_true", help="Keep R_charg/R_partial_charg/R_FC/R_other_rest columns in final outputs")
    p.add_argument("--no-fullcharge-parking-mode", action="store_true", help="Disable R_other_rest -> R_FC move-and-park merge")
    p.add_argument("--merge-gap-minutes", type=float, default=30.0)
    p.add_argument("--min-other-rest-hours", type=float, default=7.0, help="Use <=0 to disable duration condition")
    p.add_argument("--full-soc-threshold", type=float, default=95.0)
    p.add_argument("--delay-soc-threshold", type=float, default=80.0)
    p.add_argument("--departure-buffer-h", type=float, default=1.0)
    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    min_other = None if args.min_other_rest_hours <= 0 else args.min_other_rest_hours
    config = DFCConfig(
        full_soc_threshold=args.full_soc_threshold,
        delay_soc_threshold=args.delay_soc_threshold,
        departure_buffer_h=args.departure_buffer_h,
        fullcharge_parking_mode=not args.no_fullcharge_parking_mode,
        merge_gap_minutes=args.merge_gap_minutes,
        min_other_rest_hours_for_merge=min_other,
        remove_label_columns=not args.keep_labels,
    )

    process_path(
        input_path=args.input,
        output_dir=args.output_dir,
        summary_csv_path=args.summary_csv,
        config=config,
        pattern=args.pattern,
        workers=args.workers,
        skip_existing=not args.overwrite,
    )


if __name__ == "__main__":
    # On Windows, call through command line. multiprocessing works with top-level worker.
    main()
