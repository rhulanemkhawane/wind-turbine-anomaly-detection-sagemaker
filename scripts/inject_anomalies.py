"""
One-time, reproducible preprocessing step that bakes synthetic anomalies into
the raw Turkey SCADA dataset and writes a labeled version to data/labeled/.

Fault types:
  - curtailment: power forced to ~0 despite adequate wind speed (grid curtailment)
  - derating: power reduced by a flat 20% (sustained derating)

Usage:
  python scripts/inject_anomalies.py \
      --input data/raw/T1.csv \
      --output-dir data/labeled \
      --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATED_CAPACITY_KW = 3600.0
CUT_IN_SPEED_MS = 3.5

CURTAILMENT_TARGET_FRACTION = 0.05
CURTAILMENT_DURATION_RANGE = (6, 36)  # 10-min records => 1-6 hours
CURTAILMENT_MAX_FRACTION_OF_RATED = 0.05

DERATING_TARGET_FRACTION = 0.03
DERATING_DURATION_RANGE = (12, 72)  # 10-min records => 2-12 hours
DERATING_REDUCTION = 0.20


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
    df = df.sort_values("Date/Time").reset_index(drop=True)
    return df


def place_curtailment_events(wind_speed: np.ndarray, occupied: np.ndarray, rng, max_attempts=5000):
    """Pick non-overlapping windows where wind speed is mostly adequate for cut-in."""
    n = len(wind_speed)
    target_rows = int(n * CURTAILMENT_TARGET_FRACTION)
    events = []
    covered = 0
    attempts = 0
    while covered < target_rows and attempts < max_attempts:
        attempts += 1
        length = int(rng.integers(CURTAILMENT_DURATION_RANGE[0], CURTAILMENT_DURATION_RANGE[1] + 1))
        start = int(rng.integers(0, n - length))
        end = start + length
        if occupied[start:end].any():
            continue
        if (wind_speed[start:end] >= CUT_IN_SPEED_MS).mean() < 0.8:
            continue
        events.append((start, end))
        occupied[start:end] = True
        covered += length
    return events


def place_derating_events(n: int, occupied: np.ndarray, rng, max_attempts=5000):
    target_rows = int(n * DERATING_TARGET_FRACTION)
    events = []
    covered = 0
    attempts = 0
    while covered < target_rows and attempts < max_attempts:
        attempts += 1
        length = int(rng.integers(DERATING_DURATION_RANGE[0], DERATING_DURATION_RANGE[1] + 1))
        start = int(rng.integers(0, n - length))
        end = start + length
        if occupied[start:end].any():
            continue
        events.append((start, end))
        occupied[start:end] = True
        covered += length
    return events


def main(args: argparse.Namespace) -> None:
    df = load_raw(Path(args.input))
    n = len(df)
    rng = np.random.default_rng(args.seed)

    df["original_power_kw"] = df["LV ActivePower (kW)"].astype(float)
    df["is_anomaly"] = 0
    df["event_id"] = ""
    df["fault_type"] = ""

    occupied = np.zeros(n, dtype=bool)
    wind_speed = df["Wind Speed (m/s)"].to_numpy()
    power = df["LV ActivePower (kW)"].to_numpy(dtype=float)

    curtailment_events = place_curtailment_events(wind_speed, occupied, rng)
    derating_events = place_derating_events(n, occupied, rng)

    event_records = []
    event_counter = 0

    for start, end in curtailment_events:
        event_counter += 1
        eid = f"CURT-{event_counter:04d}"
        for i in range(start, end):
            if wind_speed[i] >= CUT_IN_SPEED_MS:
                power[i] = rng.uniform(0.0, CURTAILMENT_MAX_FRACTION_OF_RATED * RATED_CAPACITY_KW)
        df.loc[start : end - 1, "is_anomaly"] = 1
        df.loc[start : end - 1, "event_id"] = eid
        df.loc[start : end - 1, "fault_type"] = "curtailment"
        event_records.append(
            {
                "event_id": eid,
                "event_start": df["Date/Time"].iloc[start],
                "event_end": df["Date/Time"].iloc[end - 1],
                "fault_type": "curtailment",
                "duration_records": end - start,
            }
        )

    for start, end in derating_events:
        event_counter += 1
        eid = f"DERATE-{event_counter:04d}"
        power[start:end] = power[start:end] * (1 - DERATING_REDUCTION)
        df.loc[start : end - 1, "is_anomaly"] = 1
        df.loc[start : end - 1, "event_id"] = eid
        df.loc[start : end - 1, "fault_type"] = "derating"
        event_records.append(
            {
                "event_id": eid,
                "event_start": df["Date/Time"].iloc[start],
                "event_end": df["Date/Time"].iloc[end - 1],
                "fault_type": "derating",
                "duration_records": end - start,
            }
        )

    # Power can't go negative even after a derating cut on an already-near-zero reading.
    power = np.clip(power, 0.0, None)
    df["LV ActivePower (kW)"] = power

    events_df = pd.DataFrame(event_records).sort_values("event_start").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled_scada.csv"
    events_path = out_dir / "events.csv"
    df.to_csv(labeled_path, index=False)
    events_df.to_csv(events_path, index=False)

    curt_rows = sum(e - s for s, e in curtailment_events)
    derate_rows = sum(e - s for s, e in derating_events)
    print(f"rows total: {n}")
    print(f"curtailment events: {len(curtailment_events)}  rows: {curt_rows} ({curt_rows / n:.2%})")
    print(f"derating events:    {len(derating_events)}  rows: {derate_rows} ({derate_rows / n:.2%})")
    print(f"wrote: {labeled_path}")
    print(f"wrote: {events_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/T1.csv")
    parser.add_argument("--output-dir", default="data/labeled")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
