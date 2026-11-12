"""Download and transform NASA OMNI2 hourly ASCII records."""
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACT = ROOT / "artifacts" / "solarshield.joblib"
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
YEARS = range(2010, 2026)
WIND = ["speed", "bz", "density"]
FEATURES = ["speed_mean", "bz_mean", "bz_min", "density_mean", "southward_drive"]
FEATURE_LABELS = {
    "speed_mean": "Mean wind speed",
    "bz_mean": "Mean Bz (GSM)",
    "bz_min": "Minimum Bz (GSM)",
    "density_mean": "Mean proton density",
    "southward_drive": "Mean speed × southward Bz",
}
STEP = pd.Timedelta(hours=3)
SOURCE = "NASA OMNI2 hourly, bow-shock shifted"


def fetch(year, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"omni2_{year}.dat"
    if path.exists():
        return path
    partial = path.with_suffix(".part")
    print(f"Downloading {path.name}", flush=True)
    try:
        with urlopen(BASE_URL + path.name, timeout=45) as response, partial.open("wb") as output:
            copyfileobj(response, output)
        read_year(partial, year)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def read_year(path, year):
    # NASA's 1-based words 17, 24, 25, 39 are Bz GSM, density, speed and Kp×10.
    columns = {0: "year", 1: "day", 2: "hour", 16: "bz", 23: "density", 24: "speed", 38: "kp"}
    raw = pd.read_csv(path, sep=r"\s+", header=None, usecols=list(columns)).rename(columns=columns)
    for column in raw:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    start = pd.Timestamp(year, 1, 1, tz="UTC")
    stop = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    valid = raw.year.eq(year) & raw.day.between(1, (stop - start).days) & raw.hour.between(0, 23)
    valid &= raw.day.mod(1).eq(0) & raw.hour.mod(1).eq(0)
    if not valid.all() or raw.empty:
        raise ValueError(f"{path} has invalid OMNI2 timestamps for {year}.")
    times = start + pd.to_timedelta((raw.day - 1) * 24 + raw.hour, unit="h")
    hourly = raw[WIND + ["kp"]].set_index(pd.DatetimeIndex(times, name="time"))
    if hourly.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate hours.")
    if hourly.index.min() != start or hourly.index.max() != stop - pd.Timedelta(hours=1):
        raise ValueError(f"{path} does not span the full calendar year.")
    hourly["speed"] = hourly.speed.where(hourly.speed.gt(0) & hourly.speed.lt(9999))
    hourly["bz"] = hourly.bz.where(hourly.bz.abs().lt(999.9))
    hourly["density"] = hourly.density.where(hourly.density.ge(0) & hourly.density.lt(999.9))
    kp_codes = np.rint(np.arange(28) * (10 / 3)).astype(int)
    hourly["kp"] = (hourly.kp.where(hourly.kp.isin(kp_codes)) * 3 / 10).round() / 3
    return hourly


def make_periods(hourly):
    # UTC bins start at 00, 03, ... 21. Preserve the time grid so shifts never bridge gaps.
    hourly = hourly.sort_index().asfreq("h")
    hourly["drive"] = hourly.speed * (-hourly.bz).clip(lower=0) / 1000
    grouped = hourly.resample("3h")
    periods = grouped[WIND].mean()
    kp = grouped.kp.first()
    periods["kp"] = kp.where(grouped.kp.nunique().eq(1))
    stats = grouped.agg(
        speed_mean=("speed", "mean"), bz_mean=("bz", "mean"), bz_min=("bz", "min"),
        density_mean=("density", "mean"), southward_drive=("drive", "mean"),
    )
    # Require at least two valid hours per input, including two paired speed/Bz hours.
    usable = grouped[WIND + ["drive"]].count().ge(2).all(axis=1)
    predictors = stats.where(usable, np.nan).shift(1)
    periods = periods.join(predictors)
    periods["storm"] = periods.kp.ge(5).where(periods.kp.notna()).astype("Int8")
    return periods


def load_periods(directory, download=True):
    hourly = []
    for year in YEARS:
        path = fetch(year, directory) if download else directory / f"omni2_{year}.dat"
        hourly.append(read_year(path, year))
    return make_periods(pd.concat(hourly))
