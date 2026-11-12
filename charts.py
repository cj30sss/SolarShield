"""Plotly figures for the SolarShield dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from omni import FEATURE_LABELS, STEP, WIND

CYAN = "#55e5ff"
BLUE = "#6488ff"
PINK = "#ff749a"
WIND_LABELS = ["Wind speed · km/s", "Bz GSM · nT", "Proton density · cm⁻³"]
WIND_COLORS = [CYAN, BLUE, "#92e6b5"]


def style(fig, height):
    return fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0b1526", plot_bgcolor="#0b1526",
        font_color="#dceaff", height=height, margin=dict(l=48, r=34, t=48, b=45),
        hoverlabel=dict(bgcolor="#122540"), transition_duration=300,
    )


def timeline(data, threshold, focus=None):
    titles = WIND_LABELS + ["Observed Kp", "Model storm probability"]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.065)
    for row, (field, label, color) in enumerate(zip(WIND, WIND_LABELS, WIND_COLORS), 1):
        fig.add_scatter(x=data.index, y=data[field], name=label, line_color=color, row=row, col=1)
    fig.add_scatter(x=data.index, y=data.kp, name="Observed Kp", line=dict(color=CYAN, shape="hv"), row=4, col=1)
    storms = data[data.kp.ge(5)]
    fig.add_scatter(x=storms.index, y=storms.kp, name="Observed storm", mode="markers",
                    marker_color=PINK, row=4, col=1)
    fig.add_hline(y=5, row=4, col=1, line_dash="dot", line_color=PINK)
    fig.add_scatter(x=data.index, y=data.probability, name="Model probability",
                    line=dict(color=BLUE, shape="hv"), row=5, col=1)
    fig.add_hline(y=threshold, row=5, col=1, line_dash="dot", line_color=PINK)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.03], row=5, col=1)
    if focus is not None:
        fig.add_vline(x=focus.to_pydatetime(), line_color="#ffffff", line_dash="dash")
    fig.update_xaxes(title_text="Target interval start · UTC", row=5, col=1)
    return style(fig, 780).update_layout(hovermode="x unified", hoversubplots="axis", showlegend=False)


def gauge(score, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * score, number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": CYAN, "thickness": 0.25},
            "steps": [
                {"range": [0, threshold * 50], "color": "#183855"},
                {"range": [threshold * 50, threshold * 100], "color": "#284877"},
                {"range": [threshold * 100, 100], "color": "#563353"},
            ],
            "threshold": {"value": threshold * 100, "line": {"color": PINK, "width": 3}},
        },
    ))
    return style(fig, 285)


def feature_bars(values, axis, percent=False):
    ordered = pd.Series(values).sort_values()
    fig = go.Figure(go.Bar(
        x=ordered.values, y=[FEATURE_LABELS[key] for key in ordered.index], orientation="h",
        marker_color=[CYAN if value >= 0 else PINK for value in ordered.values],
    ))
    fig.update_xaxes(title=axis, tickformat=".1%" if percent else ".3f", zeroline=True)
    return style(fig, 300).update_layout(margin=dict(l=180, r=34, t=30, b=55))


def confusion(matrix):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Quiet", "Storm"], y=["Quiet", "Storm"], texttemplate="%{z:,}",
        colorscale=[[0, "#14253f"], [1, "#20649a"]], showscale=False,
        hovertemplate="Observed %{y}<br>Predicted %{x}<br>%{z:,} periods<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted class")
    fig.update_yaxes(title="Observed class", autorange="reversed")
    return style(fig, 320)


def precursors(data):
    # An onset needs an observed quiet interval immediately before the first storm interval.
    onsets = np.flatnonzero((data.kp.ge(5) & data.kp.shift(1).lt(5)).to_numpy())
    offsets = range(-8, 1)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=WIND_LABELS, vertical_spacing=0.11)
    for row, (field, color) in enumerate(zip(WIND, WIND_COLORS), 1):
        values = pd.concat({
            offset * int(STEP / pd.Timedelta(hours=1)): data[field].shift(-offset).iloc[onsets].reset_index(drop=True)
            for offset in offsets
        }, axis=1)
        lower, median, upper = (values.quantile(q) for q in (0.25, 0.5, 0.75))
        fig.add_scatter(x=lower.index, y=lower, line_width=0, hoverinfo="skip", row=row, col=1)
        fig.add_scatter(x=upper.index, y=upper, line_width=0, fill="tonexty",
                        fillcolor="rgba(85,229,255,.13)", hoverinfo="skip", row=row, col=1)
        fig.add_scatter(x=median.index, y=median, line_color=color, customdata=values.count(),
                        hovertemplate="%{y:.2f} · %{customdata} onsets<extra></extra>", row=row, col=1)
    fig.update_xaxes(title_text="Hours relative to onset · 0 = first storm interval", row=3, col=1)
    return style(fig, 540).update_layout(showlegend=False, hovermode="x unified"), len(onsets)
