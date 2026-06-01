#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Figure: former Fig.3 + Fig.4

Panel assignment
- A-D: former Fig.3 SOC examples
- E: former Fig.4A cluster scatter
- F-I: former Fig.4B-E t_FC histograms

Layout
- Full-width figure, based on Fig.4 width
- A/B remain on the left-top block
- C/D move to the former E position at the top-right
- E moves to the former C/D position at the left-bottom and keeps its manual size/position adjustment
- F-I remain as a 2 x 2 block below C/D

Use EMPTY_LAYOUT_MODE = True to draw only empty axes boxes for fast layout tuning.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator


# ─────────────────────────────────────────────────────
# Fast layout mode
# ─────────────────────────────────────────────────────
EMPTY_LAYOUT_MODE = False
# True  : 데이터 없이 빈 그래프 박스만 그림
# False : 실제 데이터 읽어서 최종 그림 그림

# ─────────────────────────────────────────────────────
# Global style: Fig.4 기준으로 통일
# ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 5,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
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
# Paths
# ─────────────────────────────────────────────────────
CLUSTER_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"

CSV_T95 = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95\t95_before_after_delta_combined.csv"

DIR_CR_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing",
}

DIR_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차",
}

DIR_BAD_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\불량개입",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\불량개입",
}

OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\Combined_Fig"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# Former Fig.3 panel specs → A-D
# ─────────────────────────────────────────────────────
PANEL_SPECS_SOC = {
    "A": {
        "cluster": 2,
        "base_key": "bms_01241228090_2023-04",
        "start_day": 9,
        "end_day": 15,
    },
    "B": {
        "cluster": 1,
        "base_key": "bms_01241248817_2023-04",
        "start_day": 8,
        "end_day": 14,
    },
    "C": {
        "cluster": 0,
        "base_key": "bms_altitude_01241248932_2024-05",
        "start_day": 24,
        "end_day": 30,
    },
    "D": {
        "dfc_variant": "bad",
        "base_key": "bms_01241228037_2023-04",
        "vehicle_model": "Ioniq5",
        "start_day": 4,
        "end_day": 10,
    },
}


# ─────────────────────────────────────────────────────
# Colors / styles
# ─────────────────────────────────────────────────────
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

COLOR_APPLIED = "#0073c2"
COLOR_NOT     = "#cd534c"
COLOR_DELTA   = "#efc000"
ALPHA_MAIN    = 0.450
ALPHA_INSET   = 0.450

CUTOFF_DFC_AFTER_LAST_SOC0 = True
SOC0_EPS = 1e-9

RE_BASE = re.compile(r"(bms_(?:altitude_)?\d+_\d{4}-\d{2})", re.IGNORECASE)


# ─────────────────────────────────────────────────────
# Histogram options: former Fig.4
# ─────────────────────────────────────────────────────
BIN_WIDTH_DEFAULT       = 5.0
INSET_BIN_STEP_DEFAULT  = 5.0
INSET_TICK_STEP_DEFAULT = 50.0
LABEL_EVERY             = 10

INSET_WIDTH_FRAC  = 0.40
INSET_HEIGHT_FRAC = 0.40
INSET_PAD         = 0.06
EXTRA_DOWN        = 0.00

PLOT_LIMITS = {
    "ALL": {
        "main_mode": "fixed",
        "main_x": (0, 200),
        "main_y": 200,
        "inset_mode": "fixed",
        "inset_x": (0, 150),
        "inset_y": 150,
        "bin_width": 5.0,
        "inset_bin_step": 5.0,
        "inset_tick_step": 50.0,
    },
    "cluster0": {
        "main_mode": "fixed",
        "main_x": (0, 200),
        "main_y": 200,
        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
        "bin_width": 5.0,
        "inset_bin_step": 5.0,
        "inset_tick_step": 50.0,
    },
    "cluster1": {
        "main_mode": "fixed",
        "main_x": (0, 410),
        "main_y": None,
        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
        "bin_width": 10.0,
        "inset_bin_step": 10.0,
        "inset_tick_step": 150.0,
    },
    "cluster2": {
        "main_mode": "auto",
        "main_x": None,
        "main_y": None,
        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
        "bin_width": 10.0,
        "inset_bin_step": 10.0,
        "inset_tick_step": 200.0,
    },
}


# ─────────────────────────────────────────────────────
# Common utilities
# ─────────────────────────────────────────────────────
def thin_spines(ax, lw: float = 0.4):
    for sp in ax.spines.values():
        sp.set_linewidth(lw)

def scale_axes_about_center(ax_list, scale=0.90):
    """
    각 Axes를 중심 기준으로 같은 비율로 축소.
    scale=0.90이면 가로/세로 모두 90% 크기로 줄임.
    """
    for ax in ax_list:
        pos = ax.get_position()
        new_w = pos.width * scale
        new_h = pos.height * scale
        new_x0 = pos.x0 + (pos.width - new_w) / 2
        new_y0 = pos.y0 + (pos.height - new_h) / 2
        ax.set_position([new_x0, new_y0, new_w, new_h])



def resize_axes_about_center(ax_list, width_scale=1.0, height_scale=1.0, dx=0.0, dy=0.0):
    """
    Resize/move multiple Axes about their centers.
    Use this mainly for Fig.3 SOC panels. Panel letters can be fixed
    separately using add_panel_label_figpos().
    """
    for ax in ax_list:
        pos = ax.get_position()
        new_w = pos.width * width_scale
        new_h = pos.height * height_scale
        new_x0 = pos.x0 + (pos.width - new_w) / 2 + dx
        new_y0 = pos.y0 + (pos.height - new_h) / 2 + dy
        ax.set_position([new_x0, new_y0, new_w, new_h])


def shrink_axis(ax, scale=0.90):
    """Same as the original Fig.4 helper: shrink one Axes inside its GridSpec cell."""
    pos = ax.get_position()
    new_w = pos.width * scale
    new_h = pos.height * scale
    new_x0 = pos.x0 + (pos.width - new_w) / 2
    new_y0 = pos.y0 + (pos.height - new_h) / 2
    ax.set_position([new_x0, new_y0, new_w, new_h])


def resize_axis_wh(ax, width_scale=0.90, height_scale=0.76):
    """Same as the original Fig.4 helper: resize one Axes about its center."""
    pos = ax.get_position()
    new_w = pos.width * width_scale
    new_h = pos.height * height_scale
    new_x0 = pos.x0 + (pos.width - new_w) / 2
    new_y0 = pos.y0 + (pos.height - new_h) / 2
    ax.set_position([new_x0, new_y0, new_w, new_h])

def add_panel_label(ax, label, fontsize=10, x=-0.10, y=1.04):
    ax.text(
        x, y, label.upper(),
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def add_panel_label_figpos(fig, pos, label, fontsize=10, dx=-0.018, dy=0.006):
    """
    Draw panel label in figure coordinates using a stored Axes position.
    Use this when the graph box is moved after plotting but the panel label
    should remain at the pre-move position.
    """
    fig.text(
        pos.x0 + dx,
        pos.y1 + dy,
        label.upper(),
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


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


def draw_empty_box(ax, xlabel="", ylabel="", xlim=None, ylim=None, xticks=None, yticks=None):
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlabel(xlabel, labelpad=1.0)
    ax.set_ylabel(ylabel, labelpad=1.0)
    ax.tick_params(axis="both", width=0.4, length=2.0, pad=1.0)
    ax.minorticks_off()
    thin_spines(ax, lw=0.4)


# ─────────────────────────────────────────────────────
# Former Fig.3 SOC functions
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


def load_bms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "soc" not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")
    df = df.dropna(subset=["time", "soc"]).sort_values("time")
    return df


def find_vehicle_model(df_clusters: pd.DataFrame, base_key: str, cluster: int) -> str:
    if "vehicle model" not in df_clusters.columns:
        raise ValueError("[ERR] 'vehicle model' 컬럼이 없습니다.")

    sub = df_clusters[
        (df_clusters["base_key"] == base_key) &
        (df_clusters["cluster"] == cluster)
    ]

    if sub.empty:
        sub2 = df_clusters[df_clusters["base_key"] == base_key]
        if sub2.empty:
            raise ValueError(f"[ERR] base_key={base_key} 에 해당하는 vehicle model을 찾지 못했습니다.")
        vm = str(sub2["vehicle model"].iloc[0])
    else:
        vm = str(sub["vehicle model"].iloc[0])

    return vm


def build_month_paths(vehicle_model: str, base_key: str, dfc_variant: str = "normal") -> Tuple[str, str]:
    if vehicle_model not in DIR_CR_MAP:
        raise ValueError(f"[ERR] 지원하지 않는 vehicle model: {vehicle_model}")

    cr_dir = DIR_CR_MAP[vehicle_model]

    if dfc_variant == "bad":
        if vehicle_model not in DIR_BAD_DFC_MAP:
            raise ValueError(f"[ERR] DIR_BAD_DFC_MAP에 vehicle model 없음: {vehicle_model}")
        dfc_dir = DIR_BAD_DFC_MAP[vehicle_model]
    else:
        if vehicle_model not in DIR_DFC_MAP:
            raise ValueError(f"[ERR] DIR_DFC_MAP에 vehicle model 없음: {vehicle_model}")
        dfc_dir = DIR_DFC_MAP[vehicle_model]

    cr_path = os.path.join(cr_dir, f"{base_key}_CR.csv")
    dfc_path = os.path.join(dfc_dir, f"{base_key}_DFC.csv")

    if not os.path.isfile(cr_path):
        raise FileNotFoundError(f"[ERR] CR file not found: {cr_path}")
    if not os.path.isfile(dfc_path):
        raise FileNotFoundError(f"[ERR] DFC file not found: {dfc_path}")

    return cr_path, dfc_path


def parse_start_end_from_spec(spec: dict, base_key: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if spec.get("start_ts") is not None:
        start_ts = pd.to_datetime(spec["start_ts"], errors="raise")
        if spec.get("end_ts") is None:
            end_ts = start_ts + pd.Timedelta(days=7)
        else:
            end_ts = pd.to_datetime(spec["end_ts"], errors="raise")
        return start_ts, end_ts

    try:
        ym = base_key.split("_")[-1]
        year, month = map(int, ym.split("-"))
    except Exception:
        raise ValueError(f"[ERR] base_key에서 year-month 파싱 실패: {base_key}")

    if "start_day" not in spec:
        raise ValueError("[ERR] start_day 또는 start_ts가 필요합니다.")

    start_day = int(spec["start_day"])
    start_time = str(spec.get("start_time", "00:00:00"))
    start_ts = pd.to_datetime(
        f"{year:04d}-{month:02d}-{start_day:02d} {start_time}",
        errors="raise"
    )

    if spec.get("end_ts") is not None:
        end_ts = pd.to_datetime(spec["end_ts"], errors="raise")
        return start_ts, end_ts

    if "end_day" in spec:
        end_day = int(spec["end_day"])
        end_time = str(spec.get("end_time", "23:59:59"))
        end_ts = pd.to_datetime(
            f"{year:04d}-{month:02d}-{end_day:02d} {end_time}",
            errors="raise"
        )
        return start_ts, end_ts

    end_ts = start_ts + pd.Timedelta(days=7)
    return start_ts, end_ts


def slice_and_pad_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    if end_ts <= start_ts:
        return pd.DataFrame(columns=df.columns)

    win = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()

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

    prior_end = df[df["time"] <= end_ts].tail(1)
    if not prior_end.empty:
        soc_end = prior_end["soc"].iloc[0]
    else:
        after_end = df[df["time"] >= end_ts].head(1)
        soc_end = after_end["soc"].iloc[0] if not after_end.empty else win["soc"].iloc[-1]

    if win["time"].max() < end_ts:
        win = pd.concat(
            [win, pd.DataFrame({"time": [end_ts], "soc": [soc_end]})],
            ignore_index=True
        )

    return win.sort_values("time").reset_index(drop=True)


def convert_to_day_axis(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    span = end_ts - start_ts
    if span.total_seconds() <= 0:
        df["day_idx"] = 1.0
        return df

    frac = (df["time"] - start_ts) / span
    df["day_idx"] = frac * 7.0
    return df


def style_time_soc_axes(ax):
    ax.set_xlim(0, 7)
    ax.set_xticks(np.arange(0, 8))
    ax.set_xticklabels([str(i) for i in range(0, 8)])

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_ylabel("SOC (%)", labelpad=1.0)
    ax.minorticks_off()
    ax.tick_params(axis="both", width=0.4, length=2.0, pad=1.0)
    thin_spines(ax, lw=0.4)


def draw_soc_panel(
    ax,
    df_cr: pd.DataFrame,
    df_ap: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp
):
    if df_cr.empty and df_ap.empty:
        ax.axis("off")
        return

    df_ap_plot = df_ap
    if CUTOFF_DFC_AFTER_LAST_SOC0:
        cutoff_ts = last_soc0_block_start_time(df_ap)
        if cutoff_ts is not None:
            df_ap_plot = df_ap[df_ap["time"] < cutoff_ts].copy()

    cr = convert_to_day_axis(df_cr, start_ts, end_ts)
    ap = convert_to_day_axis(df_ap_plot, start_ts, end_ts)

    ax.plot(
        cr["day_idx"], cr["soc"],
        color=CLR_CR, linestyle=LS_CR, linewidth=LW_CR,
        alpha=ALPHA_CR, label=LAB_CR,
    )

    if not ap.empty:
        ax.plot(
            ap["day_idx"], ap["soc"],
            color=CLR_APPL, linestyle=LS_APPL, linewidth=LW_APPL,
            alpha=ALPHA_APPL, label=LAB_APPL,
        )

    style_time_soc_axes(ax)

    ax.legend(
        loc="lower left",
        frameon=False,
        borderaxespad=0.2,
    )


def draw_soc_panel_from_spec(ax, label: str, spec: dict, df_clusters: pd.DataFrame):
    if EMPTY_LAYOUT_MODE:
        draw_empty_box(
            ax,
            xlabel="",
            ylabel="SOC (%)",
            xlim=(0, 7),
            ylim=(0, 100),
            xticks=np.arange(0, 8),
            yticks=[0, 20, 40, 60, 80, 100],
        )
        return

    base_key = spec["base_key"]
    dfc_variant = str(spec.get("dfc_variant", "normal")).lower()

    try:
        start_ts, end_ts = parse_start_end_from_spec(spec, base_key)

        if spec.get("vehicle_model") is not None:
            vm = str(spec["vehicle_model"])
        else:
            if "cluster" not in spec:
                raise ValueError("[ERR] cluster 또는 vehicle_model 중 하나는 필요합니다.")
            vm = find_vehicle_model(df_clusters, base_key, int(spec["cluster"]))

        cr_path, dfc_path = build_month_paths(vm, base_key, dfc_variant=dfc_variant)

        df_cr_full = load_bms_csv(cr_path)
        df_ap_full = load_bms_csv(dfc_path)

        df_cr_win = slice_and_pad_range(df_cr_full, start_ts, end_ts)
        df_ap_win = slice_and_pad_range(df_ap_full, start_ts, end_ts)

        draw_soc_panel(ax, df_cr_win, df_ap_win, start_ts, end_ts)

    except Exception as e:
        print(f"[SKIP] SOC panel {label}: {e}")
        draw_empty_box(
            ax,
            xlabel="",
            ylabel="SOC (%)",
            xlim=(0, 7),
            ylim=(0, 100),
            xticks=np.arange(0, 8),
            yticks=[0, 20, 40, 60, 80, 100],
        )


# ─────────────────────────────────────────────────────
# Former Fig.4A: cluster scatter
# ─────────────────────────────────────────────────────
def pick_feature_columns(df: pd.DataFrame):
    n_candidates = ["delta_t95_event_N", "N_events", "N_events_applied", "N_events_total"]
    mean_candidates = ["delta_t95_event_mean_h", "delta_t95_mean_h", "delayed_mean_h"]

    n_col = next((c for c in n_candidates if c in df.columns), None)
    mean_col = next((c for c in mean_candidates if c in df.columns), None)

    if n_col is None or mean_col is None:
        raise ValueError(
            "cluster scatter용 피처 컬럼이 없습니다. "
            "delta_t95_event_N / delta_t95_event_mean_h 또는 대체 후보 컬럼이 필요합니다."
        )
    return n_col, mean_col


def plot_cluster_scatter_ax(ax, df_feat: pd.DataFrame):
    if EMPTY_LAYOUT_MODE:
        draw_empty_box(
            ax,
            xlabel="N(DFC)",
            ylabel=r"AVG($\Delta t_{\mathrm{FC}}$) (hours)",
            xlim=(0, 30),
            ylim=(0, 100),
        )
        return

    df_feat = df_feat.copy()
    if "cluster" not in df_feat.columns:
        ax.axis("off")
        return

    n_col, mean_col = pick_feature_columns(df_feat)

    x_base_col = "N_used" if "N_used" in df_feat.columns else n_col
    y_base_col = "mean_used" if "mean_used" in df_feat.columns else mean_col

    x_num = pd.to_numeric(df_feat[x_base_col], errors="coerce")
    y_num = pd.to_numeric(df_feat[y_base_col], errors="coerce")
    cl_num = pd.to_numeric(df_feat["cluster"], errors="coerce")

    mask = x_num.notna() & y_num.notna() & cl_num.notna()
    used = df_feat.loc[mask].copy()
    if used.empty:
        ax.axis("off")
        return

    used[x_base_col] = pd.to_numeric(used[x_base_col], errors="coerce")
    used[y_base_col] = pd.to_numeric(used[y_base_col], errors="coerce")
    used["cluster"] = pd.to_numeric(used["cluster"], errors="coerce").astype(int)

    cluster_labels = {
        0: r"Minimal $R_{\mathrm{FC}}$",
        1: r"Frequent $R_{\mathrm{FC}}$",
        2: r"Long $R_{\mathrm{FC}}$",
    }
    palette = ["#cd534c", "#4dbbd5", "#0073c2"]

    for cid in [2, 1, 0]:
        sub = used[used["cluster"] == cid]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_base_col],
            sub[y_base_col],
            s=18,
            marker="o",
            c=palette[cid % len(palette)],
            edgecolor="k",
            linewidth=0.3,
            alpha=0.7,
            label=cluster_labels.get(cid, f"Cluster {cid}"),
        )

    x_min, x_max = used[x_base_col].min(), used[x_base_col].max()
    y_min, y_max = used[y_base_col].min(), used[y_base_col].max()

    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_xlabel("N(DFC)", fontsize=6, labelpad=1.0)
    ax.set_ylabel(r"AVG($\Delta t_{\mathrm{FC}}$) (hours)", fontsize=6, labelpad=1.0)

    ax.tick_params(axis="both", labelsize=5, width=0.4, length=2.5, pad=1.5)
    ax.minorticks_off()
    thin_spines(ax, lw=0.4)

    ax.legend(
        fontsize=5,
        loc="upper right",
        frameon=False,
    )


# ─────────────────────────────────────────────────────
# Former Fig.4B-E: histogram + inset
# ─────────────────────────────────────────────────────
def plot_overlapped_hist_with_inset_ax(ax, df_t95: pd.DataFrame, cfg_key: str = "ALL"):
    if EMPTY_LAYOUT_MODE:
        draw_empty_box(
            ax,
            xlabel=r"$t_{\mathrm{FC},m}$ (hours)",
            ylabel="Count",
            xlim=(0, 200),
            ylim=(0, 200),
        )

        # 빈 inset 박스도 같이 표시
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(0.54, 0.50, 0.40, 0.40),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        draw_empty_box(
            axins,
            xlabel=r"$\Delta t_{\mathrm{FC},m}$ (hours)",
            ylabel="",
            xlim=(0, 150),
            ylim=(0, 150),
            xticks=[0, 50, 100, 150],
            yticks=[],
        )
        return

    cfg = PLOT_LIMITS[cfg_key]

    main_mode = cfg["main_mode"]
    main_x = cfg["main_x"]
    main_y = cfg["main_y"]

    inset_mode = cfg["inset_mode"]
    inset_x = cfg["inset_x"]
    inset_y = cfg["inset_y"]

    bin_w = cfg.get("bin_width", BIN_WIDTH_DEFAULT)
    inset_bin_step = cfg.get("inset_bin_step", INSET_BIN_STEP_DEFAULT)
    inset_tick_step = cfg.get("inset_tick_step", INSET_TICK_STEP_DEFAULT)

    usecols = ["t95_before_h", "t95_after_h", "delta_t_h"]
    for c in usecols:
        if c not in df_t95.columns:
            raise ValueError(f"Missing column: {c}")

    after = pd.to_numeric(df_t95["t95_after_h"], errors="coerce").dropna().to_numpy()
    before = pd.to_numeric(df_t95["t95_before_h"], errors="coerce").dropna().to_numpy()

    if after.size == 0 and before.size == 0:
        ax.axis("off")
        return

    all_data = np.concatenate([after, before])
    dmin, dmax = float(all_data.min()), float(all_data.max())

    left = np.floor(dmin / bin_w) * bin_w
    right = np.ceil(dmax / bin_w) * bin_w

    if main_mode == "fixed" and main_x is not None:
        left = min(left, main_x[0])
        right = max(right, np.ceil(main_x[1] / bin_w) * bin_w)

    edges = np.arange(left, right + bin_w * 0.999, bin_w)
    centers = (edges[:-1] + edges[1:]) / 2

    cnt_after, _ = np.histogram(after, bins=edges)
    cnt_before, _ = np.histogram(before, bins=edges)

    h_non = ax.bar(
        centers,
        cnt_before,
        width=bin_w,
        color=COLOR_NOT,
        alpha=ALPHA_MAIN,
        edgecolor="k",
        linewidth=0.3,
        label="non-DFC",
    )
    h_dfc = ax.bar(
        centers,
        cnt_after,
        width=bin_w,
        color=COLOR_APPLIED,
        alpha=ALPHA_MAIN,
        edgecolor="k",
        linewidth=0.3,
        label="DFC",
    )

    ax.set_ylabel("Count", labelpad=1.0)
    ax.set_xlabel(r"$t_{\mathrm{FC},m}$ (hours)", labelpad=1.0)

    peak = max(cnt_after.max(), cnt_before.max())

    if main_mode == "fixed" and main_x is not None and main_y is not None:
        ax.set_xlim(*main_x)
        ax.set_ylim(0, main_y)
    else:
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, peak * 1.15)

    tick_positions = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1e-9, bin_w)
    tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
    ax.set_xticks(tick_positions[tick_idx])
    ax.set_xticklabels([str(int(x)) for x in tick_positions[tick_idx]])

    ax.minorticks_off()
    ax.grid(False)

    leg = ax.legend(
        handles=[h_non[0], h_dfc[0]],
        labels=["non-DFC", "DFC"],
        loc="upper right",
        frameon=False,
    )

    ax.tick_params(axis="both", width=0.4, pad=1.5)
    thin_spines(ax, lw=0.4)

    delta = pd.to_numeric(df_t95["delta_t_h"], errors="coerce").dropna().to_numpy()
    if delta.size == 0:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bbox = leg.get_window_extent(renderer=renderer)

    x_disp = ax.bbox.x1
    y_disp = leg_bbox.y0
    _, y0_ax = ax.transAxes.inverted().transform((x_disp, y_disp))

    x0 = 1 - INSET_WIDTH_FRAC - INSET_PAD
    y0 = max(INSET_PAD, y0_ax - INSET_HEIGHT_FRAC - INSET_PAD - EXTRA_DOWN)

    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(x0, y0, INSET_WIDTH_FRAC, INSET_HEIGHT_FRAC),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )

    if inset_mode == "fixed" and inset_x is not None:
        lo, hi = inset_x
        lo_edge = np.floor(lo / inset_bin_step) * inset_bin_step
        hi_edge = np.ceil(hi / inset_bin_step) * inset_bin_step
    else:
        dmin2, dmax2 = delta.min(), delta.max()
        lo_edge = np.floor(dmin2 / inset_bin_step) * inset_bin_step
        hi_edge = np.ceil(dmax2 / inset_bin_step) * inset_bin_step
        lo, hi = lo_edge, hi_edge

    bins_dt = np.arange(lo_edge, hi_edge + inset_bin_step * 0.999, inset_bin_step)

    axins.hist(
        delta,
        bins=bins_dt,
        color=COLOR_DELTA,
        alpha=ALPHA_INSET,
        edgecolor="k",
        linewidth=0.3,
    )

    axins.set_xlim(lo, hi)

    if inset_mode == "fixed" and inset_y is not None:
        axins.set_ylim(0, inset_y)

    axins.set_xlabel(r"$\Delta t_{\mathrm{FC},m}$ (hours)", labelpad=0.5)
    axins.tick_params(axis="both", width=0.4, labelsize=5, pad=1.0)
    axins.xaxis.set_major_locator(MultipleLocator(inset_tick_step))
    axins.minorticks_off()
    thin_spines(axins, lw=0.4)


# ─────────────────────────────────────────────────────
# Data loading for former Fig.4
# ─────────────────────────────────────────────────────
def load_fig4_data():
    t95 = pd.read_csv(CSV_T95)
    cl = pd.read_csv(CLUSTER_CSV)

    t95 = ensure_base_key_col(t95)
    cl = ensure_base_key_col(cl)

    cl["cluster"] = pd.to_numeric(cl["cluster"], errors="coerce")
    merged = pd.merge(t95, cl[["base_key", "cluster"]], how="left", on="base_key")

    sub0 = merged[merged["cluster"] == 0]
    sub1 = merged[merged["cluster"] == 1]
    sub2 = merged[merged["cluster"] == 2]

    return merged, sub0, sub1, sub2, cl


# ─────────────────────────────────────────────────────
# Main combined figure
# ─────────────────────────────────────────────────────
def make_combined_figure():
    """
    Combined figure with Fig.3 placed above the original Fig.4 layout.

    Panel assignment:
      A-D: former Fig.3 SOC examples, arranged as 2 x 2.
      E-I: former Fig.4 panels, with the original Fig.4 layout preserved:
           E = original Fig.4A scatter, F-I = original Fig.4B-E histograms.

    Important:
      - Fig.4 internal GridSpec, axis resizing, and label placement are kept
        the same as the provided Fig.4 code as much as possible.
      - A-D panel labels are fixed using fig.text, so they do not move when
        the A-D graph boxes are resized or shifted.
    """
    if EMPTY_LAYOUT_MODE:
        df_clusters = pd.DataFrame()
        merged = sub0 = sub1 = sub2 = pd.DataFrame()
        df_feat = pd.DataFrame()
    else:
        df_clusters = pd.read_csv(CLUSTER_CSV)
        df_clusters = ensure_base_key_col(df_clusters)
        df_clusters["cluster"] = pd.to_numeric(df_clusters["cluster"], errors="coerce")
        merged, sub0, sub1, sub2, df_feat = load_fig4_data()

    # ─────────────────────────────────────────────────────
    # Layout control knobs
    # ─────────────────────────────────────────────────────
    # Full-width figure. Width is the original Fig.4 width; height is expanded
    # only to place Fig.3 above Fig.4.
    FIG_W = 7.24
    FIG_H = 6.20

    # Space between Fig.3 block and Fig.4 block.
    # Decrease to bring Fig.4 closer to Fig.3.
    OUTER_HSPACE = 0.19

    # Fig.3-only graph-box adjustment.
    # These affect A-D graph boxes only. A-D panel letters stay fixed.
    FIG3_WIDTH_SCALE = 0.87
    FIG3_HEIGHT_SCALE = 0.96
    FIG3_DX = -0.010
    FIG3_DY = 0.000

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)

    # Top = Fig.3, Bottom = original Fig.4 layout.
    outer = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[1.10, 1.18],
        hspace=OUTER_HSPACE,
        figure=fig,
    )

    # ─────────────────────────────────────────────────────
    # Top block: former Fig.3, 2 x 2
    # ─────────────────────────────────────────────────────
    gs_fig3 = GridSpecFromSubplotSpec(
        nrows=2,
        ncols=2,
        subplot_spec=outer[0, 0],
        wspace=0.10,
        hspace=0.40,
    )

    axA = fig.add_subplot(gs_fig3[0, 0])
    axB = fig.add_subplot(gs_fig3[0, 1], sharex=axA, sharey=axA)
    axC = fig.add_subplot(gs_fig3[1, 0], sharex=axA, sharey=axA)
    axD = fig.add_subplot(gs_fig3[1, 1], sharex=axA, sharey=axA)

    soc_axes = {"A": axA, "B": axB, "C": axC, "D": axD}
    for label, ax in soc_axes.items():
        spec = PANEL_SPECS_SOC[label]
        draw_soc_panel_from_spec(ax, label, spec, df_clusters)

    # Right SOC panels: remove y-axis label only, keep tick labels.
    axB.set_ylabel("")
    axD.set_ylabel("")

    # Bottom row only gets x labels.
    axC.set_xlabel("Time (days)", labelpad=1.0)
    axD.set_xlabel("Time (days)", labelpad=1.0)

    # Apply normal margins first so Axes positions are final enough to store labels.
    # Values are close to the original Fig.4, with more height available.
    fig.subplots_adjust(
        left=0.060,
        right=0.980,
        bottom=0.070,
        top=0.965,
    )

    # Store A-D label positions BEFORE moving/resizing A-D graph boxes.
    # This is what prevents A-D panel letters from moving.
    label_pos_ABCD = {label: ax.get_position() for label, ax in soc_axes.items()}

    # Resize/move Fig.3 graph boxes only. A-D letters are not tied to the axes.
    resize_axes_about_center(
        [axA, axB, axC, axD],
        width_scale=FIG3_WIDTH_SCALE,
        height_scale=FIG3_HEIGHT_SCALE,
        dx=FIG3_DX,
        dy=FIG3_DY,
    )

    # Fixed A-D panel labels in figure coordinates.
    for label, old_pos in label_pos_ABCD.items():
        add_panel_label_figpos(
            fig,
            old_pos,
            label,
            fontsize=10,
            dx=-0.018,
            dy=0.006,
        )

    # ─────────────────────────────────────────────────────
    # Bottom block: original Fig.4 layout preserved
    # Original Fig.4 layout:
    #   - 2 rows x 3 columns
    #   - left column spans both rows for cluster scatter
    #   - right 2 x 2 block for histograms
    #   - width_ratios=[1.7, 1, 1], wspace=0.18, hspace=0.22
    #   - shrink b-e with scale=0.86
    #   - resize scatter with width_scale=0.82, height_scale=0.74
    # ─────────────────────────────────────────────────────
    gs_fig4 = GridSpecFromSubplotSpec(
        nrows=2,
        ncols=3,
        subplot_spec=outer[1, 0],
        width_ratios=[1.7, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.22,
    )

    # Original Fig.4A -> combined panel E
    axE = fig.add_subplot(gs_fig4[:, 0])
    plot_cluster_scatter_ax(axE, df_feat)

    # Original Fig.4B-E -> combined panels F-I
    axF = fig.add_subplot(gs_fig4[0, 1])
    axG = fig.add_subplot(gs_fig4[0, 2])
    axH = fig.add_subplot(gs_fig4[1, 1])
    axI = fig.add_subplot(gs_fig4[1, 2])

    plot_overlapped_hist_with_inset_ax(axF, merged, cfg_key="ALL")

    if EMPTY_LAYOUT_MODE or len(sub2) > 0:
        plot_overlapped_hist_with_inset_ax(axG, sub2, cfg_key="cluster2")

    if EMPTY_LAYOUT_MODE or len(sub1) > 0:
        plot_overlapped_hist_with_inset_ax(axH, sub1, cfg_key="cluster1")

    if EMPTY_LAYOUT_MODE or len(sub0) > 0:
        plot_overlapped_hist_with_inset_ax(axI, sub0, cfg_key="cluster0")

    # Fig.4 label cleanup requested for the combined layout.
    # - G/I: remove main y-axis label only.
    # - F/G: remove main x-axis label only.
    # Tick labels remain unchanged.
    axG.set_ylabel("")
    axI.set_ylabel("")
    axF.set_xlabel("")
    axG.set_xlabel("")

    # Keep original Fig.4 axis resizing behavior.
    for ax in [axF, axG, axH, axI]:
        shrink_axis(ax, scale=0.86)

    resize_axis_wh(axE, width_scale=0.82, height_scale=0.74)

    # Keep original Fig.4 label style, only relabeled E-I.
    add_panel_label(axE, "E", x=-0.08, y=1.02)
    add_panel_label(axF, "F", x=-0.12, y=1.08)
    add_panel_label(axG, "G", x=-0.12, y=1.08)
    add_panel_label(axH, "H", x=-0.12, y=1.08)
    add_panel_label(axI, "I", x=-0.12, y=1.08)

    suffix = "layout_empty" if EMPTY_LAYOUT_MODE else "final"
    base = os.path.join(OUT_DIR, f"Figure_2_{suffix}")

    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)

    plt.close(fig)
    print(f"[SAVE] {base}.png / {base}.pdf")


if __name__ == "__main__":
    make_combined_figure()