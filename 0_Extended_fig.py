#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import calendar
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────
# Global style (Nature Energy-like)  ─ (원본 Fig.3 스타일 유지)
# ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 5,          # base
    "axes.labelsize": 5,
    "axes.titlesize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
})

# ─────────────────────────────────────────────────────
# 경로/설정
# ─────────────────────────────────────────────────────
CLUSTER_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"

# 정상 DFC용: Long / Frequent / Minimal
DIR_CR_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차",
}
DIR_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차",
}

# 불량개입용: 기존 불량개입 코드에서 쓰던 경로
DIR_BAD_CR_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing",
}
DIR_BAD_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\불량개입",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\불량개입",
}

# 저장 폴더
BASE_OUT = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\Suppl"
os.makedirs(BASE_OUT, exist_ok=True)

# ─────────────────────────────────────────────────────
# 4행×3열 = 12 panels
#
# a b c = Long 의 a,d,g
# d e f = Frequent 의 a,d,g
# g h i = Minimal 의 a,d,g
# j k l = 불량개입 의 a,b,c
#
# 중요:
# - 기존 코드 구조 그대로 쓰기 위해 불량개입도 start/end 토큰 형식으로 넣음
# - 불량개입만 "kind": "bad" 로 표시해서 파일 경로만 다르게 읽음
# ─────────────────────────────────────────────────────
PANEL_SPECS_12 = [
    # a-c: Long a,d,g
    {"kind": "normal", "base_key": "bms_01241228082_2023-04", "start": "24", "end": "1"},
    {"kind": "normal", "base_key": "bms_01241248909_2023-02", "start": "21", "end": "28"},
    {"kind": "normal", "base_key": "bms_01241248913_2023-09", "start": "20", "end": "27"},

    # d-f: Frequent a,d,g
    {"kind": "normal", "base_key": "bms_01241228055_2023-05", "start": "8",  "end": "15"},
    {"kind": "normal", "base_key": "bms_01241248924_2023-03", "start": "15", "end": "22"},
    {"kind": "normal", "base_key": "bms_01241228095_2023-10", "start": "8",  "end": "15"},

    # g-i: Minimal a,d,g
    {"kind": "normal", "base_key": "bms_01241248780_2023-02", "start": "21",    "end": "28"},
    {"kind": "normal", "base_key": "bms_altitude_01241597802_2023-12", "start": "22:18", "end": "29:18"},
    {"kind": "normal", "base_key": "bms_01241228132_2023-01", "start": "1:14",  "end": "8:14"},

    # j-l: 불량개입 a,b,c
    {"kind": "bad", "base_key": "bms_01241228086_2023-10", "start": "16", "end": "23"},
    {"kind": "bad", "base_key": "bms_01241228095_2023-01", "start": "6",  "end": "13"},
    {"kind": "bad", "base_key": "bms_01241228097_2023-07", "start": "23", "end": "30"},
]

# 색상/스타일 (R vs DFC) — 기존 그대로
CLR_CR      = "#cd534c"
LS_CR       = "-"
CLR_APPL    = "#0073c2"
LS_APPL     = "--"
LW_CR       = 1.0
LW_APPL     = 1.0
ALPHA_CR    = 0.9
ALPHA_APPL  = 0.9
LAB_CR      = "non-DFC"
LAB_APPL    = "DFC"

# legend 표시 방식: "first"(첫 패널만) or "each"(모든 패널)
LEGEND_MODE = "each"

RE_BASE = re.compile(r"(bms_(?:altitude_)?\d+_\d{4}-\d{2})", re.IGNORECASE)

# 불량개입 코드에서 쓰던 cutoff 옵션
CUTOFF_DFC_AFTER_LAST_SOC0 = True
SOC0_EPS = 1e-9


# ─────────────────────────────────────────────────────
# 유틸 함수 — 기존 코드 유지
# ─────────────────────────────────────────────────────
def ensure_base_key_col(df: pd.DataFrame) -> pd.DataFrame:
    if "base_key" in df.columns:
        return df
    keys = []
    for _, row in df.iterrows():
        k = None
        for val in row.values:
            if isinstance(val, str):
                m = RE_BASE.search(val)
                if m:
                    k = m.group(1)
                    break
        keys.append(k)
    df = df.copy()
    df["base_key"] = keys
    return df


def load_bms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "soc" not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")
    df = df.dropna(subset=["time", "soc"]).sort_values("time")
    return df


def thin_spines(ax, lw: float = 0.4):
    for sp in ax.spines.values():
        sp.set_linewidth(lw)


def style_time_soc_axes(ax):
    ax.set_xlim(0, 7)
    ax.set_xticks(np.arange(0, 8))
    ax.set_xticklabels([str(i) for i in range(0, 8)])

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlabel("Time (day)", labelpad=1.0)
    ax.set_ylabel("SOC (%)", labelpad=1.0)

    ax.tick_params(axis="both", width=0.4, length=2.0, pad=1.0)
    thin_spines(ax, lw=0.4)


def convert_to_day_axis(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    span = (end_ts - start_ts)
    if span.total_seconds() <= 0:
        df["day_idx"] = 1.0
        return df
    frac = (df["time"] - start_ts) / span
    df["day_idx"] = frac * 7.0
    return df


def slice_and_pad_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    if end_ts <= start_ts:
        return pd.DataFrame(columns=df.columns)

    win = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()

    # start padding
    prior_start = df[df["time"] <= start_ts].tail(1)
    if not prior_start.empty:
        soc_start = prior_start["soc"].iloc[0]
    else:
        after_start = df[df["time"] >= start_ts].head(1)
        if after_start.empty:
            return pd.DataFrame(columns=df.columns)
        soc_start = after_start["soc"].iloc[0]

    if win.empty or win["time"].min() > start_ts:
        win = pd.concat(
            [pd.DataFrame({"time": [start_ts], "soc": [soc_start]}), win],
            ignore_index=True
        )

    # end padding
    prior_end = df[df["time"] <= end_ts].tail(1)
    if not prior_end.empty:
        soc_end = prior_end["soc"].iloc[0]
    else:
        after_end = df[df["time"] >= end_ts].head(1)
        soc_end = after_end["soc"].iloc[0] if not after_end.empty else win["soc"].iloc[-1]

    if not win.empty and win["time"].max() < end_ts:
        win = pd.concat(
            [win, pd.DataFrame({"time": [end_ts], "soc": [soc_end]})],
            ignore_index=True
        )

    return win.sort_values("time").reset_index(drop=True)


def add_panel_label(ax, label, fontsize=7):
    ax.text(
        -0.08, 1.02, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False
    )


def parse_day_hour_token(tok: str) -> Tuple[int, str]:
    """
    "2:18" -> (2, "18:00:00")
    "8"    -> (8, "00:00:00")
    """
    s = str(tok).strip()
    if ":" in s:
        day_str, hour_str = s.split(":", 1)
        day = int(day_str)
        hour = int(hour_str)
        return day, f"{hour:02d}:00:00"
    return int(s), "00:00:00"


def get_month_end_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_start_end_from_tokens(base_key: str, start_tok: str, end_tok: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    규칙:
      - start/end가 "D" 또는 "D:H" 형태
      - end_day < start_day 이면 "월말까지"로 해석 (예: 24,1)
      - end 시간 미지정이면 기본 23:59:59(정수 day) / "D:H"면 H:00:00
    """
    ym = base_key.split("_")[-1]
    year, month = map(int, ym.split("-"))

    s_day, s_time = parse_day_hour_token(start_tok)
    end_has_hour = (":" in str(end_tok).strip())
    e_day, e_time_tmp = parse_day_hour_token(end_tok)

    start_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{s_day:02d} {s_time}")

    if e_day < s_day:
        last_day = get_month_end_day(year, month)
        end_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{last_day:02d} 23:59:59")
        return start_ts, end_ts

    if end_has_hour:
        end_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{e_day:02d} {e_time_tmp}")
    else:
        end_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{e_day:02d} 23:59:59")

    return start_ts, end_ts


# ─────────────────────────────────────────────────────
# 불량개입용 SOC=0 cutoff 함수만 추가
# ─────────────────────────────────────────────────────
def last_soc0_block_start_time(
    df: pd.DataFrame,
    soc_col: str = "soc",
    time_col: str = "time"
) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    if soc_col not in df.columns or time_col not in df.columns:
        return None

    soc = pd.to_numeric(df[soc_col], errors="coerce")
    mask = soc.notna() & (soc <= SOC0_EPS)
    if not mask.any():
        return None

    arr = mask.to_numpy()
    last_idx = int(np.flatnonzero(arr)[-1])
    start_idx = last_idx
    while start_idx > 0 and arr[start_idx - 1]:
        start_idx -= 1

    return pd.to_datetime(df.iloc[start_idx][time_col])


# ─────────────────────────────────────────────────────
# 파일 경로 함수
# 기존 build_month_paths 구조 유지 + kind만 받아서 경로만 분기
# ─────────────────────────────────────────────────────
def build_month_paths(vehicle_model: str, base_key: str, kind: str = "normal") -> Tuple[str, str]:
    if kind == "bad":
        if vehicle_model not in DIR_BAD_CR_MAP:
            raise ValueError(f"[ERR] 지원하지 않는 vehicle model: {vehicle_model}")

        cr_dir = DIR_BAD_CR_MAP[vehicle_model]
        dfc_dir = DIR_BAD_DFC_MAP[vehicle_model]

        r_path = os.path.join(cr_dir, f"{base_key}_CR.csv")
        dfc_path = os.path.join(dfc_dir, f"{base_key}_DFC.csv")

        if not os.path.isfile(r_path):
            raise FileNotFoundError(f"[ERR] bad non-DFC file not found: {r_path}")
        if not os.path.isfile(dfc_path):
            raise FileNotFoundError(f"[ERR] bad DFC file not found: {dfc_path}")

        return r_path, dfc_path

    # normal
    if vehicle_model not in DIR_CR_MAP:
        raise ValueError(f"[ERR] 지원하지 않는 vehicle model: {vehicle_model}")

    r_dir = DIR_CR_MAP[vehicle_model]
    dfc_dir = DIR_DFC_MAP[vehicle_model]

    r_path_primary = os.path.join(r_dir, f"{base_key}_R.csv")
    r_path_fallback = os.path.join(r_dir, f"{base_key}_CR.csv")

    if os.path.isfile(r_path_primary):
        r_path = r_path_primary
    elif os.path.isfile(r_path_fallback):
        r_path = r_path_fallback
    else:
        raise FileNotFoundError(
            f"[ERR] non-DFC file not found: {r_path_primary} (또는 {r_path_fallback})"
        )

    dfc_path = os.path.join(dfc_dir, f"{base_key}_DFC.csv")
    if not os.path.isfile(dfc_path):
        raise FileNotFoundError(f"[ERR] DFC file not found: {dfc_path}")

    return r_path, dfc_path


def resolve_vehicle_model(base_key: str, cl: Optional[pd.DataFrame], kind: str = "normal") -> str:
    """
    1) cluster CSV에 base_key가 있으면 vehicle model 사용
    2) 없으면 EV6/Ioniq5 경로에서 파일 존재 여부로 판별
    """
    if cl is not None and "vehicle model" in cl.columns and "base_key" in cl.columns:
        sub = cl[cl["base_key"] == base_key]
        if not sub.empty:
            return str(sub["vehicle model"].iloc[0])

    hits = []

    if kind == "bad":
        for vm in ("EV6", "Ioniq5"):
            r = os.path.join(DIR_BAD_CR_MAP[vm], f"{base_key}_CR.csv")
            d = os.path.join(DIR_BAD_DFC_MAP[vm], f"{base_key}_DFC.csv")
            if os.path.isfile(r) and os.path.isfile(d):
                hits.append(vm)
    else:
        for vm in ("EV6", "Ioniq5"):
            r_dir = DIR_CR_MAP[vm]
            dfc_dir = DIR_DFC_MAP[vm]

            r1 = os.path.join(r_dir, f"{base_key}_R.csv")
            r2 = os.path.join(r_dir, f"{base_key}_CR.csv")
            d = os.path.join(dfc_dir, f"{base_key}_DFC.csv")

            has_r = os.path.isfile(r1) or os.path.isfile(r2)
            if has_r and os.path.isfile(d):
                hits.append(vm)

    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise FileNotFoundError(
            f"[ERR] base_key={base_key}: EV6/Ioniq5 어디에서도 파일 쌍을 찾지 못함"
        )
    raise RuntimeError(
        f"[ERR] base_key={base_key}: EV6/Ioniq5 양쪽에서 파일이 발견됨(애매함). 수동 지정 필요: {hits}"
    )


def draw_panel(ax, df_r: pd.DataFrame, df_dfc: pd.DataFrame,
               start_ts: pd.Timestamp, end_ts: pd.Timestamp,
               show_legend: bool):
    if df_r.empty and df_dfc.empty:
        ax.axis("off")
        return

    r = convert_to_day_axis(df_r, start_ts, end_ts)
    d = convert_to_day_axis(df_dfc, start_ts, end_ts)

    ax.plot(
        r["day_idx"], r["soc"],
        color=CLR_CR,
        linestyle=LS_CR,
        linewidth=LW_CR,
        alpha=ALPHA_CR,
        label=LAB_CR
    )
    ax.plot(
        d["day_idx"], d["soc"],
        color=CLR_APPL,
        linestyle=LS_APPL,
        linewidth=LW_APPL,
        alpha=ALPHA_APPL,
        label=LAB_APPL
    )

    style_time_soc_axes(ax)
    ax.tick_params(axis="x", labelbottom=True)

    if show_legend:
        leg = ax.legend(loc="lower left", frameon=True, framealpha=0.9, borderaxespad=0.2)
        frame = leg.get_frame()
        frame.set_edgecolor("grey")
        frame.set_linewidth(0.4)


# ─────────────────────────────────────────────────────
# 4×3 full-width Figure 생성
# 기존 코드 구성 그대로 유지
# ─────────────────────────────────────────────────────
def make_suppl_long_4users_3weeks_fullwidth():
    cl = None
    if os.path.isfile(CLUSTER_CSV):
        try:
            tmp = pd.read_csv(CLUSTER_CSV)
            tmp = ensure_base_key_col(tmp)
            cl = tmp
        except Exception:
            cl = None

    fig, axes = plt.subplots(
        nrows=4, ncols=3,
        figsize=(7.09, 5.00),  # 기존 비율 유지
        dpi=300,
        sharex=True, sharey=True,
    )
    axes = axes.ravel()
    panel_labels = list("abcdefghijkl")

    for i, (spec, lab) in enumerate(zip(PANEL_SPECS_12, panel_labels)):
        ax = axes[i]
        base_key = spec["base_key"]
        kind = spec.get("kind", "normal")

        try:
            start_ts, end_ts = parse_start_end_from_tokens(base_key, spec["start"], spec["end"])
        except Exception as e:
            print(f"[SKIP] panel {lab}: time parse error - {e}")
            ax.axis("off")
            add_panel_label(ax, lab)
            continue

        try:
            vm = resolve_vehicle_model(base_key, cl, kind=kind)
        except Exception as e:
            print(f"[SKIP] panel {lab}: vehicle model resolve error - {e}")
            ax.axis("off")
            add_panel_label(ax, lab)
            continue

        try:
            r_path, dfc_path = build_month_paths(vm, base_key, kind=kind)
        except Exception as e:
            print(f"[SKIP] panel {lab}: file path error - {e}")
            ax.axis("off")
            add_panel_label(ax, lab)
            continue

        try:
            df_r_full = load_bms_csv(r_path)
            df_d_full = load_bms_csv(dfc_path)
        except Exception as e:
            print(f"[SKIP] panel {lab}: load error - {e}")
            ax.axis("off")
            add_panel_label(ax, lab)
            continue

        # 불량개입 패널만 기존 불량개입 코드처럼 SOC=0 이후 DFC curve cutoff
        if kind == "bad" and CUTOFF_DFC_AFTER_LAST_SOC0:
            cutoff_ts = last_soc0_block_start_time(df_d_full)
            if cutoff_ts is not None:
                df_d_full = df_d_full[df_d_full["time"] < cutoff_ts].copy()

        df_r_win = slice_and_pad_range(df_r_full, start_ts, end_ts)
        df_d_win = slice_and_pad_range(df_d_full, start_ts, end_ts)

        if df_r_win.empty and df_d_win.empty:
            print(f"[SKIP] panel {lab}: no data in selected range.")
            ax.axis("off")
            add_panel_label(ax, lab)
            continue

        show_legend = (LEGEND_MODE == "each") or (LEGEND_MODE == "first" and i == 0)
        draw_panel(ax, df_r_win, df_d_win, start_ts, end_ts, show_legend=show_legend)
        add_panel_label(ax, lab)

    for j in range(len(PANEL_SPECS_12), len(axes)):
        axes[j].axis("off")

    for ax in axes[:len(PANEL_SPECS_12)]:
        ax.tick_params(axis="y", labelleft=True)

    fig.tight_layout(rect=[0.04, 0.03, 0.995, 0.99])

    out_base = os.path.join(BASE_OUT, "Suppl_Fig_Long_Frequent_Minimal_Bad")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {out_base}.png / {out_base}.pdf")


if __name__ == "__main__":
    make_suppl_long_4users_3weeks_fullwidth()