# SolarShield

SolarShield is a local Streamlit mission-control dashboard for live NOAA observations and retrospective geomagnetic-storm prediction. The training command downloads the NASA dataset and saves a reusable model locally.

## Run locally

Open a terminal in the `SolarShield` folder. Use Python 3.11 or newer. The project was verified with Python 3.12 and scikit-learn 1.9.1.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell, then run the same three `python` commands. Open the local address printed by Streamlit, usually [http://localhost:8501](http://localhost:8501).

`python train.py` downloads any missing annual files, verifies them, trains all three classifiers, prints the metrics and saves `artifacts/solarshield.joblib`. Launching the dashboard after that reuses the artifact. Run training again if you install a different scikit-learn version. Only load trusted joblib artifacts.

### Data options

- To download and verify without training, run `python train.py --download-only`.
- For a manual import, put `omni2_2010.dat` through `omni2_2025.dat` in `data/`. Use the original annual OMNI2 ASCII files from the [NASA SPDF archive](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/). Run `python train.py --offline`.
- Use `--data-dir /path/to/files` for a different source folder. The dashboard reads the default model artifact. Use `--output` if you need a separate copy.
- If a cached annual file is incomplete or corrupt, remove that file and run training again to fetch a new copy.
- Without an artifact, the NOAA and spacecraft pages still open. Without network access, the historical pages still work with a saved artifact. The NOAA page reports missing measurements and marks readings more than 20 minutes old as stale.

## How it works

- **Inputs and labels.** NASA OMNI2 hourly values are parsed using the [documented column layout and fill values](https://omniweb.gsfc.nasa.gov/html/ow_data.html). UTC bins begin at 00:00, 03:00, and so on. A repeated, consistent Kp value labels the entire bin. Kp ≥ 5 marks a storm period, including ongoing storms. Bins without a known label are excluded from evaluation.
- **Timing.** For a target beginning at time *t*, the model uses only solar-wind observations in `[t−3h, t)`. Inputs are mean speed, mean and minimum GSM Bz, mean proton density, and mean speed multiplied by the southward component of Bz (scaled by 1,000). Each input requires at least two valid hours. The combined feature requires at least two paired speed/Bz observations. Kp serves as the label only. Missing wind periods are skipped and gaps are preserved on the time grid.
- **Training and validation.** Models fit 2010–2018 with inverse-frequency sample weights. Average precision (the stepwise summary of the precision–recall curve) on 2019–2021 selects the winner. A sigmoid fitted on validation scores maps each model’s output to a probability estimate. Validation also sets each high-band cutoff by maximizing F1. The moderate band starts at half that cutoff. These are dashboard display and classification thresholds. The validation confusion matrices use their respective cutoffs.
- **Test.** Only the selected model is evaluated on 2022–2025. Its model, calibration and cutoff are frozen before evaluation. The dashboard identifies training and validation selections when replaying dates. Those periods were used to fit or select components of the model.
- **Live compatibility.** OMNI2 is a processed, merged historical product whose solar-wind timestamps are shifted to Earth’s bow shock. The [NOAA SWPC real-time feeds](https://www.spaceweather.gov/products/solar-wind) provide upstream, in-situ minute measurements and NOAA also publishes estimated Kp. Units and GSM coordinates match, while source processing, timing, and data latency differ. NOAA observations stay in the live panel and are never sent to the OMNI2 model. The historical exercise does not validate operational warning lead time. [NOAA alerts](https://www.spaceweather.gov/products/alerts-watches-and-warnings) are published separately.

## Project files

| File | Purpose |
| --- | --- |
| `omni.py` | NASA download, import, quality checks and three-hour features |
| `train.py` | Chronological training, calibration, selection, evaluation and persistence |
| `noaa.py` | Current NOAA feeds, latest valid observations and error handling |
| `app.py` | Dashboard, historical prediction and storm replay |
| `charts.py` | Interactive wind and Kp timelines, gauge, feature charts and confusion matrices |
| `.streamlit/config.toml` | Dark dashboard theme |
