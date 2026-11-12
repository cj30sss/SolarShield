"""Streamlit mission control for SolarShield."""
from time import sleep

import joblib
import pandas as pd
import streamlit as st

from charts import confusion, feature_bars, gauge, precursors, timeline
from noaa import fetch_live
from omni import ARTIFACT, FEATURES, SOURCE, STEP
from train import probability

st.set_page_config(page_title="SolarShield · Mission Control", page_icon="☉", layout="wide")

st.markdown("""
<style>
.stApp {
  background: radial-gradient(ellipse at 85% 0%, #152949 0, transparent 47%),
              linear-gradient(165deg, #070c18, #060c19 70%);
}
.block-container { max-width: 1280px; padding-top: 2.8rem; }
.hero {
  position: relative; overflow: hidden; isolation: isolate;
  border: 1px solid #26405b; border-radius: 20px; margin-bottom: 1.15rem;
  padding: clamp(1.25rem, 5vw, 3.7rem);
  background: radial-gradient(ellipse at 85% 25%, #184b69, transparent 48%),
              linear-gradient(115deg, #0c182b 20%, #112741);
}
.hero::before {
  content: ""; position: absolute; z-index: -1; inset: 0;
  background-image: radial-gradient(circle, #75dcff70 1px, transparent 1px);
  background-size: 31px 31px; mask-image: linear-gradient(90deg, transparent 30%, black);
}
.hero::after {
  content: ""; position: absolute; z-index: -1; pointer-events: none;
  right: 4%; top: -54%; width: 360px; aspect-ratio: 1; border-radius: 50%;
  border: 1px solid #66d9ff90;
  box-shadow: inset 0 0 55px #38b9ff55, 0 0 35px #38b9ff55, 0 0 0 22px #53cfff18;
  animation: orbit-glow 6s ease-in-out infinite alternate;
}
@keyframes orbit-glow { from { opacity: .45; } to { opacity: .95; } }
@media (prefers-reduced-motion: reduce) { .hero::after { animation: none; } }
.hero h1 { color: #f4faff; font-size: clamp(2.7rem, 7vw, 5.2rem); line-height: 1;
           letter-spacing: -.075em; margin: .4rem 0 .65rem; padding: 0; }
.hero p { color: #c3d9eb; margin: 0; font-size: 1.06rem; max-width: 36ch; }
.mission { color: #55e5ff; text-transform: uppercase; letter-spacing: .23em;
           font: 600 .72rem monospace; }
[data-testid="stMetric"] {
  border: 1px solid #243952; background: #0d192c; border-radius: 12px; padding: 14px;
}
[data-testid="stMetricValue"] { color: #9cecff; }
[data-testid="stBaseButton-primary"] { color: #071422; }
</style>
<div class="hero">
  <div class="mission">Space Weather Mission Control</div>
  <h1>SolarShield</h1>
  <p>Track the solar wind and replay geomagnetic storms with NASA observations.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


@st.cache_data(ttl=60, show_spinner=False)
def live_snapshot():
    values, warnings = fetch_live()
    return values, warnings, pd.Timestamp.now(tz="UTC")


def utc(time):
    return time.strftime("%Y-%m-%d %H:%M UTC")


@st.fragment(run_every="60s")
def live_panel():
    top, action = st.columns([4, 1])
    top.subheader("Live space weather")
    if action.button("Refresh NOAA feeds"):
        live_snapshot.clear()
    with st.spinner("Contacting NOAA SWPC…"):
        measurements, warnings, checked = live_snapshot()
    st.caption(f"NOAA Space Weather Prediction Center · Last checked {utc(checked)}")
    for column, (key, label, unit) in zip(st.columns(4), [
        ("proton_speed", "Solar-wind speed", "km/s"), ("bz_gsm", "Bz · GSM", "nT"),
        ("proton_density", "Proton density", "cm⁻³"), ("estimated_kp", "Estimated Kp", ""),
    ]):
        if key not in measurements:
            column.metric(label, "Unavailable")
            continue
        value, timestamp, source = measurements[key]
        age = pd.Timestamp.now(tz="UTC") - timestamp
        stale = " · stale" if age > pd.Timedelta(minutes=20) else ""
        decimals = 1 if key == "proton_speed" else 2
        column.metric(label, f"{value:,.{decimals}f} {unit}".strip())
        column.caption(f"{utc(timestamp)}{stale}" + (f" · {source}" if source else ""))
        if key == "estimated_kp":
            condition = "Storm level" if value >= 5 else "Below storm level"
            column.caption(f"At observation · {condition}")
    for warning in warnings:
        st.warning(warning)
    st.info(
        "NOAA reports near-real-time upstream solar-wind measurements and estimated Kp. "
        "SolarShield scores NASA OMNI2 hourly observations that have been merged, processed "
        "and shifted to Earth’s bow shock. Matching units and GSM coordinates leave a timing "
        "and processing mismatch, so the dashboard displays NOAA measurements separately."
    )
    st.markdown("[Official NOAA alerts and warnings](https://www.spaceweather.gov/products/alerts-watches-and-warnings)")


def split_label(time):
    if time.year <= 2018:
        return "Training period"
    if time.year <= 2021:
        return "Validation period"
    return "Held-out test period"


def prediction_panel(bundle):
    data = bundle["periods"]
    st.subheader("AI storm prediction")
    st.caption("Historical OMNI2 scoring · Kp ≥ 5 during the next 3-hour UTC interval")
    day = st.date_input(
        "Historical target date (UTC)", value=pd.Timestamp("2024-05-11").date(),
        min_value=data.index.min().date(), max_value=data.index.max().date(),
    )
    choices = data.loc[str(day)].dropna(subset=FEATURES).index
    if choices.empty:
        st.warning("No intervals on this date have enough measurements in the preceding three hours.")
        return
    target = st.selectbox("Target interval start (UTC)", choices, format_func=lambda date: date.strftime("%H:%M"))
    st.caption(f"Input window · {utc(target - STEP)} → {utc(target)} · {split_label(target)}")
    if st.button("Generate historical prediction", type="primary"):
        st.session_state.scored_period = target
    if st.session_state.get("scored_period") != target:
        return

    model, calibrator = bundle["model"], bundle["calibrator"]
    features = data.loc[[target], FEATURES]
    score = float(probability(model, calibrator, features)[0])
    threshold = bundle["threshold"]
    band = "High" if score >= threshold else "Moderate" if score >= threshold / 2 else "Low"
    left, right = st.columns([1, 1.4])
    left.metric("Model probability band", band)
    left.plotly_chart(gauge(score, threshold), config={"displayModeBar": False})
    left.caption(f"Moderate ≥ {threshold / 2:.1%} · High ≥ {threshold:.1%}")
    observed = data.at[target, "kp"]
    left.caption(f"Observed Kp · {observed:.2f}" if pd.notna(observed) else "Observed Kp unavailable")

    # Perturb one feature at a time to show the signed change relative to a training median.
    scenarios = pd.concat([features] * len(FEATURES), ignore_index=True)
    for row, feature in enumerate(FEATURES):
        scenarios.at[row, feature] = bundle["reference"][feature]
    differences = score - probability(model, calibrator, scenarios)
    right.markdown("#### What moved this prediction")
    right.plotly_chart(feature_bars(dict(zip(FEATURES, differences)), "Probability change · percentage points", True))
    right.caption("Each bar compares the observed input with its training median while the other inputs stay fixed. "
                  "Correlated features can overlap, so the bars are sensitivity checks and do not add up.")
    st.caption("Retrospective model estimate. Official warnings come from NOAA SWPC.")


def history_panel(bundle):
    data = bundle["periods"]
    threshold = bundle["threshold"]
    st.subheader("Historical analysis")
    mode = st.radio("View", ["Date range", "Storm replay", "Before storms"], horizontal=True)
    if mode == "Before storms":
        st.markdown("#### Solar wind around observed storm onset")
        chart, count = precursors(data.loc["2022":"2025"])
        st.plotly_chart(chart)
        st.caption(f"{count:,} observed onsets in the held-out years. Onset follows a known quiet interval. "
                   "Lines show medians and shaded areas show the middle 50%. Hover for coverage.")
        return
    if mode == "Date range":
        dates = st.date_input(
            "Date range (UTC)", [pd.Timestamp("2024-05-08").date(), pd.Timestamp("2024-05-15").date()],
            min_value=data.index.min().date(), max_value=data.index.max().date(),
        )
        if len(dates) < 2:
            st.info("Select an end date to display the timeline.")
            return
        view = data.loc[str(dates[0]):str(dates[1])]
        st.plotly_chart(timeline(view, threshold))
        st.caption("Pink markers show observed storms. Dotted lines show Kp 5 and the model’s high-band cutoff. "
                   "Wind traces are observed in the displayed interval. Each model estimate uses the previous interval.")
        return

    onsets = data.index[data.kp.ge(5) & data.kp.shift(1).lt(5)]
    if onsets.empty:
        st.info("No observed storm onsets are available in this dataset.")
        return
    nearest = int(abs(onsets - pd.Timestamp("2024-05-10 18:00", tz="UTC")).argmin())
    event = st.selectbox("Observed storm onset", onsets, index=nearest, format_func=utc)
    view = data.loc[event - 8 * STEP:event + 8 * STEP]
    start, end = int((view.index[0] - event) / STEP), int((view.index[-1] - event) / STEP)
    cursor_key = f"cursor_{event}"
    play_key = f"play_{event}"
    play = st.button("Play storm replay", key=play_key)
    if play or cursor_key not in st.session_state:
        st.session_state[cursor_key] = 3 * end if play else 0
    offset = st.slider("Replay cursor · hours from onset", 3 * start, 3 * end, step=3, key=cursor_key)
    output = st.empty()
    # Render each step with the history available through the replay cursor.
    offsets = range(3 * start, 3 * end + 1, 3) if play else [offset]
    for frame in offsets:
        focus = view.index[int(frame) // 3 - start]
        figure = timeline(view.loc[:focus], threshold, focus)
        figure.update_xaxes(range=[view.index[0], view.index[-1]])
        output.plotly_chart(figure, key=f"replay_{event}_{frame}_{play}")
        if play and frame < 3 * end:
            sleep(0.25)
    st.caption(f"{utc(focus)} · {split_label(focus)} · Dashed line marks the replay cursor.")


def performance_panel(bundle):
    st.subheader("Model performance")
    test = bundle["test"]
    st.markdown(f"#### {bundle['winner']} · Held-out test, 2022–2025")
    for column, name in zip(st.columns(5), ["Precision", "Recall", "F1", "Accuracy", "PR-AUC (AP)"]):
        column.metric(name, f"{test[name]:.1%}")
    st.caption(f"{test['Storms']:,} storms among {test['Periods']:,} scored intervals "
               f"({test['Storms'] / test['Periods']:.1%}). Storm cutoff {test['Threshold']:.1%}.")
    left, right = st.columns(2)
    left.markdown("#### Test confusion matrix")
    left.plotly_chart(confusion(test["Confusion matrix"]))
    right.markdown("#### Feature importance")
    right.plotly_chart(feature_bars(bundle["importance"], "Validation AP decrease after shuffling"))
    right.caption("Mean of five shuffles per feature. Correlated inputs can share importance.")
    st.markdown("#### Model comparison · Validation, 2019–2021")
    columns = ["PR-AUC (AP)", "Accuracy", "Precision", "Recall", "F1", "Threshold"]
    table = pd.DataFrame(bundle["validation"]).T[columns].astype(float)
    st.dataframe(table.sort_values(columns[0], ascending=False).style.format("{:.3f}"), width="stretch")
    choice = st.selectbox("Validation confusion matrix", list(bundle["validation"]),
                          index=list(bundle["validation"]).index(bundle["winner"]))
    st.plotly_chart(confusion(bundle["validation"][choice]["Confusion matrix"]))
    st.caption("Validation average precision selects the model. Validation also sets calibration and thresholds, "
               "so the comparison metrics are descriptive. Only the selected model was evaluated on the test years.")
    with st.expander("Data coverage"):
        st.dataframe(pd.DataFrame(bundle["coverage"]).T, width="stretch")


def mission_panel():
    st.subheader("Spacecraft impacts")
    st.markdown("""
- **Orbital drag.** Geomagnetic storms heat and expand the upper atmosphere. Satellites in low Earth orbit can lose altitude faster, and operators may need to update orbit predictions or maneuver.
- **Spacecraft charging.** Changes in the charged-particle environment can cause electrical discharges that interfere with electronics.
- **Navigation and tracking.** Disturbances in the ionosphere can degrade GNSS positioning and radio signals used for tracking.
    """)
    st.markdown("[NOAA satellite drag](https://www.spaceweather.gov/impacts/satellite-drag) · "
                "[NASA spacecraft charging](https://www.nasa.gov/centers-and-facilities/nesc/understanding-the-potential-dangers-of-spacecraft-charging/) · "
                "[NOAA space weather scales](https://www.spaceweather.gov/noaa-scales-explanation)")
    st.subheader("Prediction timing")
    st.write("Each retrospective estimate uses the completed three-hour solar-wind window immediately before "
             "the target UTC interval. Historical OMNI2 data are available after processing. The experiment "
             "does not establish when equivalent inputs would have been available in real time or validate "
             "three hours of operational warning. The storm label includes both new and ongoing storms.")
    st.markdown("[NASA OMNI2 format and processing](https://omniweb.gsfc.nasa.gov/html/ow_data.html) · "
                "[NOAA real-time solar wind](https://www.spaceweather.gov/products/solar-wind)")


bundle = None
artifact_error = None
try:
    bundle = load_model(ARTIFACT, ARTIFACT.stat().st_mtime_ns)
    if bundle.get("version") != 1 or bundle.get("source") != SOURCE or bundle.get("features") != FEATURES:
        raise ValueError("Incompatible model artifact")
except FileNotFoundError:
    artifact_error = "Train the model with `python train.py` from the SolarShield folder."
except Exception as error:
    bundle = None
    artifact_error = f"The saved model could not be read ({type(error).__name__}). Run `python train.py` again."

section = st.segmented_control(
    "Mission section", ["Live NOAA", "AI Prediction", "Historical Analysis", "Model Performance", "Spacecraft Impacts"],
    default="Live NOAA", key="mission_section", label_visibility="collapsed",
)
if section == "Live NOAA":
    live_panel()
elif section == "Spacecraft Impacts":
    mission_panel()
elif section and bundle is None:
    st.info(artifact_error)
elif section == "AI Prediction":
    prediction_panel(bundle)
elif section == "Historical Analysis":
    history_panel(bundle)
elif section == "Model Performance":
    performance_panel(bundle)
