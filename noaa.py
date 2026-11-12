"""Read live SWPC measurements for display independently of the OMNI model."""
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import urlopen

import numpy as np
import pandas as pd

BASE = "https://services.swpc.noaa.gov/json/"
FEEDS = {
    "Solar-wind plasma": ("rtsw/rtsw_wind_1m.json", ("proton_speed", "proton_density")),
    "Magnetic field": ("rtsw/rtsw_mag_1m.json", ("bz_gsm",)),
    "Estimated Kp": ("planetary_k_index_1m.json", ("estimated_kp",)),
}


def latest_values(rows, fields, now):
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Empty response")
    frame["time_tag"] = pd.to_datetime(frame["time_tag"], utc=True, errors="coerce", format="mixed")
    frame = frame[frame.time_tag.le(now + pd.Timedelta(minutes=2))]
    if "active" in frame:
        frame = frame[frame.active.eq(True)]
    result = {}
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        valid = np.isfinite(values)
        if field == "estimated_kp":
            valid &= values.between(0, 9)
        elif field == "bz_gsm":
            valid &= values.abs().lt(999.9)
        elif field == "proton_speed":
            valid &= values.gt(0) & values.lt(9999)
        else:
            valid &= values.ge(0) & values.lt(999.9)
        records = frame.loc[valid].sort_values("time_tag")
        if not records.empty:
            row = records.iloc[-1]
            result[field] = (float(row[field]), row.time_tag, row.get("source"))
    return result


def fetch_live():
    def read(feed):
        name, (path, fields) = feed
        try:
            with urlopen(BASE + path, timeout=7) as response:
                rows = json.load(response)
            values = latest_values(rows, fields, pd.Timestamp.now(tz="UTC"))
            missing = [field.replace("_", " ") for field in fields if field not in values]
            warning = f"No usable {', '.join(missing)}" if missing else None
            return name, values, warning
        except (OSError, ValueError, KeyError, TypeError):
            return name, {}, "Feed unavailable"

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(read, FEEDS.items()))
    measurements = {key: value for _, values, _ in results for key, value in values.items()}
    warnings = [f"{name} · {warning}" for name, _, warning in results if warning]
    return measurements, warnings
