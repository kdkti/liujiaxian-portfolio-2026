from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RAW_CSV = Path(
    r"C:\Users\liajiaxian\Documents\Codex\2026-10-31\ge\work"
    r"\taobao_public_data\UserBehavior.csv"
)
OUT_DIR = Path(
    r"C:\Users\liajiaxian\Documents\Codex\2026-10-31\ge\outputs"
    r"\产品作品集_客户跟进与需求洞察工作台\role_portfolios\data"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
BEHAVIORS = ("pv", "cart", "fav", "buy")
BEHAVIOR_BITS = {"pv": 1, "cart": 2, "fav": 4, "buy": 8}
MAX_ID = 6_000_000
CHUNK_SIZE = 5_000_000
SAMPLE_MOD = 2000
SAMPLE_SEED_MULTIPLIERS = {
    "user": np.uint64(73856093),
    "item": np.uint64(19349663),
    "time": np.uint64(83492791),
    "behavior": np.uint64(2654435761),
}

DAY_START = int(
    datetime(2017, 11, 25, tzinfo=timezone(timedelta(hours=8))).timestamp()
)
DAY_END = int(
    datetime(2017, 12, 4, tzinfo=timezone(timedelta(hours=8))).timestamp()
)


def pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def sample_mask(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    timestamps: np.ndarray,
    behavior_codes: np.ndarray,
) -> np.ndarray:
    hashed = (
        user_ids.astype(np.uint64) * SAMPLE_SEED_MULTIPLIERS["user"]
        ^ item_ids.astype(np.uint64) * SAMPLE_SEED_MULTIPLIERS["item"]
        ^ timestamps.astype(np.uint64) * SAMPLE_SEED_MULTIPLIERS["time"]
        ^ behavior_codes.astype(np.uint64) * SAMPLE_SEED_MULTIPLIERS["behavior"]
    )
    return (hashed % np.uint64(SAMPLE_MOD)) == np.uint64(0)


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(RAW_CSV)

    behavior_events = Counter({name: 0 for name in BEHAVIORS})
    invalid_behavior = 0
    missing_cells = 0
    duplicate_rows = 0
    raw_rows = 0
    valid_rows = 0
    bad_timestamp_rows = 0

    user_flags = np.zeros(MAX_ID + 1, dtype=np.uint8)
    user_total = np.zeros(MAX_ID + 1, dtype=np.int32)
    user_buy_count = np.zeros(MAX_ID + 1, dtype=np.int32)
    first_day = np.full(MAX_ID + 1, 255, dtype=np.uint8)
    last_buy_day = np.zeros(MAX_ID + 1, dtype=np.uint8)
    user_day = np.zeros((MAX_ID + 1, 9), dtype=bool)
    user_buy_day = np.zeros((MAX_ID + 1, 9), dtype=bool)
    seen_items = np.zeros(MAX_ID + 1, dtype=bool)
    seen_categories = np.zeros(MAX_ID + 1, dtype=bool)

    day_events = np.zeros(9, dtype=np.int64)
    hour_events = np.zeros(24, dtype=np.int64)
    day_behavior = {
        name: np.zeros(9, dtype=np.int64) for name in BEHAVIORS
    }
    category_counts = {name: Counter() for name in BEHAVIORS}
    buy_item_counts: Counter[int] = Counter()
    sample_frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        RAW_CSV,
        header=None,
        names=COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "category_id": "int32",
            "behavior_type": "string",
            "timestamp": "int64",
        },
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        raw_rows += len(chunk)
        missing_cells += int(chunk.isna().sum().sum())
        duplicate_rows += int(chunk.duplicated().sum())

        timestamps = chunk["timestamp"].to_numpy(dtype=np.int64, copy=False)
        valid_time = (timestamps >= DAY_START) & (timestamps < DAY_END)
        behavior_values = chunk["behavior_type"].fillna("").to_numpy(dtype=str)
        valid_behavior = np.isin(behavior_values, BEHAVIORS)

        bad_timestamp_rows += int((~valid_time).sum())
        invalid_behavior += int((~valid_behavior).sum())

        clean = chunk.loc[valid_time & valid_behavior].copy()
        valid_rows += len(clean)
        if clean.empty:
            continue

        user_ids = clean["user_id"].to_numpy(dtype=np.int64, copy=False)
        item_ids = clean["item_id"].to_numpy(dtype=np.int64, copy=False)
        category_ids = clean["category_id"].to_numpy(dtype=np.int64, copy=False)
        timestamps = clean["timestamp"].to_numpy(dtype=np.int64, copy=False)
        behaviors = clean["behavior_type"].to_numpy(dtype=str)
        days = ((timestamps - DAY_START) // 86400).astype(np.uint8)
        hours = (((timestamps - DAY_START) // 3600) % 24).astype(np.uint8)

        seen_items[item_ids] = True
        seen_categories[category_ids] = True
        user_total += np.bincount(user_ids, minlength=MAX_ID + 1).astype(
            np.int32
        )
        user_day[user_ids, days] = True
        day_events += np.bincount(days, minlength=9)
        hour_events += np.bincount(hours, minlength=24)

        np.minimum.at(first_day, user_ids, days)

        behavior_codes = np.zeros(len(clean), dtype=np.uint8)
        for code, behavior_name in enumerate(BEHAVIORS, start=1):
            mask = behaviors == behavior_name
            behavior_codes[mask] = code
            count = int(mask.sum())
            behavior_events[behavior_name] += count
            if count:
                day_behavior[behavior_name] += np.bincount(
                    days[mask], minlength=9
                )
                selected_users = np.unique(user_ids[mask])
                user_flags[selected_users] |= BEHAVIOR_BITS[behavior_name]
                category_counts[behavior_name].update(
                    clean.loc[mask, "category_id"].value_counts().to_dict()
                )

            if behavior_name == "buy" and count:
                buy_users = user_ids[mask]
                buy_days = days[mask]
                user_buy_count += np.bincount(
                    buy_users, minlength=MAX_ID + 1
                ).astype(np.int32)
                user_buy_day[buy_users, buy_days] = True
                np.maximum.at(last_buy_day, buy_users, buy_days)
                buy_item_counts.update(
                    clean.loc[mask, "item_id"].value_counts().to_dict()
                )

        sampled_mask = sample_mask(
            user_ids, item_ids, timestamps, behavior_codes
        )
        if sampled_mask.any():
            sample_frames.append(clean.loc[sampled_mask].copy())

        print(
            f"chunk={chunk_number} raw={raw_rows:,} "
            f"valid={valid_rows:,} sampled={sum(map(len, sample_frames)):,}",
            flush=True,
        )

    if raw_rows == 0:
        raise RuntimeError("No rows were read from the source CSV.")

    active_users = np.flatnonzero(user_total > 0)
    active_count = int(len(active_users))

    sample = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else pd.DataFrame(columns=COLUMNS)
    )
    sample = sample.drop_duplicates().sort_values(
        ["timestamp", "user_id", "item_id", "behavior_type"]
    )
    sample_path = OUT_DIR / "taobao_public_sample_50k.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    daily_rows = []
    for day_index in range(9):
        date_value = datetime.fromtimestamp(
            DAY_START + day_index * 86400,
            tz=timezone(timedelta(hours=8)),
        ).date()
        active = int(user_day[active_users, day_index].sum())
        buyers = int(user_buy_day[active_users, day_index].sum())
        daily_rows.append(
            {
                "date": str(date_value),
                "events": int(day_events[day_index]),
                "active_users": active,
                "buyers": buyers,
                "buyer_rate": pct(buyers, active),
                "pv": int(day_behavior["pv"][day_index]),
                "cart": int(day_behavior["cart"][day_index]),
                "fav": int(day_behavior["fav"][day_index]),
                "buy": int(day_behavior["buy"][day_index]),
            }
        )
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")

    pv_users = int(np.count_nonzero(user_flags[active_users] & 1))
    interest_users = int(
        np.count_nonzero(user_flags[active_users] & (2 | 4))
    )
    buyer_users = int(np.count_nonzero(user_flags[active_users] & 8))
    only_pv_users = int(
        np.count_nonzero((user_flags[active_users] & 15) == 1)
    )
    funnel = pd.DataFrame(
        [
            {"stage": "全部活跃用户", "users": active_count},
            {"stage": "浏览用户 PV", "users": pv_users},
            {"stage": "收藏或加购用户", "users": interest_users},
            {"stage": "购买用户", "users": buyer_users},
        ]
    )
    funnel["rate_from_previous"] = [
        1.0,
        pct(pv_users, active_count),
        pct(interest_users, pv_users),
        pct(buyer_users, interest_users),
    ]
    funnel["rate_from_pv"] = [
        None,
        1.0,
        pct(interest_users, pv_users),
        pct(buyer_users, pv_users),
    ]
    funnel.to_csv(
        OUT_DIR / "behavior_funnel.csv", index=False, encoding="utf-8-sig"
    )

    user_activity = user_total[active_users]
    activity_bins = np.select(
        [
            user_activity == 1,
            user_activity <= 5,
            user_activity <= 20,
            user_activity <= 100,
        ],
        ["1", "2-5", "6-20", "21-100"],
        default="100+",
    )
    activity_order = ["1", "2-5", "6-20", "21-100", "100+"]
    activity_summary = (
        pd.DataFrame(
            {
                "activities_per_user": activity_bins,
                "events": user_activity,
            }
        )
        .groupby("activities_per_user", sort=False)
        .agg(users=("events", "size"), events=("events", "sum"))
        .reindex(activity_order, fill_value=0)
        .reset_index()
    )
    activity_summary["user_share"] = activity_summary["users"].map(
        lambda value: pct(value, active_count)
    )
    activity_summary["event_share"] = activity_summary["events"].map(
        lambda value: pct(value, valid_rows)
    )
    activity_summary.to_csv(
        OUT_DIR / "user_activity_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible_d1 = (first_day[active_users] <= 7) & (
        first_day[active_users] < 255
    )
    d1_users = active_users[eligible_d1]
    d1_retained = int(
        user_day[d1_users, first_day[d1_users].astype(int) + 1].sum()
    )

    eligible_d7 = (first_day[active_users] <= 1) & (
        first_day[active_users] < 255
    )
    d7_users = active_users[eligible_d7]
    d7_retained = int(
        user_day[d7_users, first_day[d7_users].astype(int) + 7].sum()
    )

    retention_rows = []
    for day_index in range(8):
        cohort = active_users[first_day[active_users] == day_index]
        if not len(cohort):
            continue
        retained = int(user_day[cohort, day_index + 1].sum())
        retention_rows.append(
            {
                "cohort_date": str(
                    datetime.fromtimestamp(
                        DAY_START + day_index * 86400,
                        tz=timezone(timedelta(hours=8)),
                    ).date()
                ),
                "cohort_users": int(len(cohort)),
                "next_day_retained": retained,
                "next_day_rate": pct(retained, len(cohort)),
            }
        )
    retention = pd.DataFrame(retention_rows)
    retention.to_csv(
        OUT_DIR / "retention_cohorts.csv", index=False, encoding="utf-8-sig"
    )

    buyer_mask = user_buy_count[active_users] > 0
    buyer_ids = active_users[buyer_mask]
    buy_frequency = user_buy_count[buyer_ids]
    recency_days = 8 - last_buy_day[buyer_ids].astype(int)
    r_bucket = np.select(
        [recency_days <= 2, recency_days <= 5],
        ["近0-2天", "近3-5天"],
        default="近6-8天",
    )
    f_bucket = np.select(
        [buy_frequency == 1, buy_frequency <= 4],
        ["购买1次", "购买2-4次"],
        default="购买5次以上",
    )
    rf_table = (
        pd.crosstab(
            pd.Series(f_bucket, name="购买频次"),
            pd.Series(r_bucket, name="最近购买"),
        )
        .reindex(
            index=["购买1次", "购买2-4次", "购买5次以上"],
            columns=["近0-2天", "近3-5天", "近6-8天"],
            fill_value=0,
        )
        .reset_index()
    )
    rf_table.to_csv(
        OUT_DIR / "rf_buyer_segments.csv", index=False, encoding="utf-8-sig"
    )

    top_category_rows = []
    for behavior_name in ("pv", "buy"):
        total_behavior = behavior_events[behavior_name]
        for category_id, count in category_counts[behavior_name].most_common(10):
            top_category_rows.append(
                {
                    "behavior": behavior_name,
                    "category_id": int(category_id),
                    "events": int(count),
                    "share": pct(count, total_behavior),
                }
            )
    top_categories = pd.DataFrame(top_category_rows)
    top_categories.to_csv(
        OUT_DIR / "top_categories.csv", index=False, encoding="utf-8-sig"
    )

    top_buy_items = pd.DataFrame(
        [
            {"item_id": int(item_id), "buy_events": int(count)}
            for item_id, count in buy_item_counts.most_common(10)
        ]
    )
    top_buy_items["share"] = top_buy_items["buy_events"].map(
        lambda value: pct(value, behavior_events["buy"])
    )
    top_buy_items.to_csv(
        OUT_DIR / "top_buy_items.csv", index=False, encoding="utf-8-sig"
    )

    hour_rows = [
        {
            "hour": hour,
            "events": int(hour_events[hour]),
            "share": pct(hour_events[hour], valid_rows),
        }
        for hour in range(24)
    ]
    hourly = pd.DataFrame(hour_rows)
    hourly.to_csv(
        OUT_DIR / "hourly_activity.csv", index=False, encoding="utf-8-sig"
    )

    all_user_paths = Counter(
        (int(flag) for flag in user_flags[active_users] if flag > 0)
    )
    path_labels = {
        1: "仅浏览",
        2: "仅加购",
        3: "浏览+加购",
        4: "仅收藏",
        5: "浏览+收藏",
        6: "加购+收藏",
        7: "浏览+加购+收藏",
        8: "仅购买",
        9: "浏览+购买",
        10: "加购+购买",
        11: "浏览+加购+购买",
        12: "收藏+购买",
        13: "浏览+收藏+购买",
        14: "加购+收藏+购买",
        15: "浏览+加购+收藏+购买",
    }
    dominant_paths = [
        {
            "path_code": flag,
            "path": path_labels[flag],
            "users": int(count),
            "share": pct(count, active_count),
        }
        for flag, count in all_user_paths.most_common(8)
    ]

    date_min = datetime.fromtimestamp(
        DAY_START, tz=timezone(timedelta(hours=8))
    ).isoformat()
    date_max = datetime.fromtimestamp(
        DAY_END - 1, tz=timezone(timedelta(hours=8))
    ).isoformat()

    funnel_records = funnel.to_dict(orient="records")
    funnel_records[0]["rate_from_pv"] = None

    summary = {
        "dataset_name": "阿里巴巴天池 Taobao UserBehavior",
        "source_url": "https://tianchi.aliyun.com/dataset/649",
        "mirror_url": "https://www.kaggle.com/datasets/marwa80/userbehavior",
        "license": "CC BY-NC-SA 4.0",
        "accessed_at": "2026-10-31",
        "analysis_type": "个人独立公开数据分析，不代表淘宝或任何公司的真实经营项目",
        "cleaning": {
            "raw_rows": raw_rows,
            "valid_rows": valid_rows,
            "bad_timestamp_rows": bad_timestamp_rows,
            "invalid_behavior_rows": invalid_behavior,
            "missing_cells": missing_cells,
            "duplicate_rows_within_chunks": duplicate_rows,
            "valid_window": [date_min, date_max],
            "timezone": "UTC+08:00",
        },
        "coverage": {
            "users": active_count,
            "items": int(seen_items.sum()),
            "categories": int(seen_categories.sum()),
            "interactions": valid_rows,
        },
        "behavior_events": {
            name: {
                "events": int(behavior_events[name]),
                "share": pct(behavior_events[name], valid_rows),
            }
            for name in BEHAVIORS
        },
        "funnel": funnel_records,
        "retention": {
            "d1_eligible_users": int(len(d1_users)),
            "d1_retained_users": d1_retained,
            "d1_rate": pct(d1_retained, len(d1_users)),
            "d7_eligible_users": int(len(d7_users)),
            "d7_retained_users": d7_retained,
            "d7_rate": pct(d7_retained, len(d7_users)),
        },
        "daily_metrics": daily.to_dict(orient="records"),
        "hourly_peak": {
            "hour": int(hourly.loc[hourly["events"].idxmax(), "hour"]),
            "events": int(hourly["events"].max()),
            "share": float(hourly["share"].max()),
        },
        "activity_segments": activity_summary.to_dict(orient="records"),
        "rf_buyer_segments": rf_table.to_dict(orient="records"),
        "top_categories": top_categories.to_dict(orient="records"),
        "top_buy_items": top_buy_items.to_dict(orient="records"),
        "dominant_user_paths": dominant_paths,
        "notes": [
            "全量汇总基于清洗后的有效行为记录计算。",
            "行为漏斗按用户是否在观察窗口内出现相应行为定义，不代表严格时间顺序。",
            "数据集没有价格和成本字段，因此只做 RF 用户分层，不计算金额 M。",
            "留存采用自然日口径，7 日留存仅纳入首日出现在 11 月 25 日或 26 日的用户。",
            "公开样本使用确定性哈希抽样，抽样模数为 2000，可按脚本复现。",
            "结果描述观察窗口内的相关关系，不构成因果结论。",
        ],
    }
    summary_path = OUT_DIR / "taobao_analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(sample_path)
    print(summary_path)
    print(
        json.dumps(
            {
                "cleaning": summary["cleaning"],
                "coverage": summary["coverage"],
                "retention": summary["retention"],
                "hourly_peak": summary["hourly_peak"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
