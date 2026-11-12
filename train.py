"""Train, select and save a reproducible retrospective SolarShield model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from omni import ARTIFACT, DATA, FEATURES, SOURCE, load_periods

SEED = 42
SPLITS = {"Train": (2010, 2018), "Validation": (2019, 2021), "Test": (2022, 2025)}


def candidates():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=8, max_depth=12, n_jobs=2, random_state=SEED,
        )),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=SEED,
        )),
    }


def raw_log_odds(model, features):
    probability = model.predict_proba(features)[:, 1].clip(1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def probability(model, calibrator, features):
    return calibrator.predict_proba(raw_log_odds(model, features))[:, 1]


def scores(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "PR-AUC (AP)": float(average_precision_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "Confusion matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "Periods": int(len(labels)),
        "Storms": int(np.count_nonzero(labels)),
        "Threshold": float(threshold),
    }


def train(periods):
    eligible = periods.dropna(subset=FEATURES + ["storm"])
    sets = {
        name: eligible[eligible.index.year.isin(range(start, end + 1))]
        for name, (start, end) in SPLITS.items()
    }
    for name, data in sets.items():
        if data.storm.nunique() != 2:
            raise ValueError(f"{name} needs usable examples of both storm and quiet periods.")
    training, validation = sets["Train"], sets["Validation"]
    weights = compute_sample_weight("balanced", training.storm)
    fitted = {}
    comparison = {}
    for name, model in candidates().items():
        model.fit(training[FEATURES], training.storm.astype(int), **{
            f"{model.steps[-1][0]}__sample_weight": weights,
        })
        # Fit a probability calibration on the naturally imbalanced validation period.
        val_logits = raw_log_odds(model, validation[FEATURES])
        calibrator = LogisticRegression(C=1e6).fit(val_logits, validation.storm.astype(int))
        val_prob = probability(model, calibrator, validation[FEATURES])
        precision, recall, thresholds = precision_recall_curve(validation.storm, val_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)
        threshold = float(thresholds[np.argmax(f1[:-1])])
        comparison[name] = scores(validation.storm, val_prob, threshold)
        comparison[name]["PR-AUC (AP)"] = float(average_precision_score(
            validation.storm, val_logits.ravel(),
        ))
        fitted[name] = (model, calibrator, threshold)

    winner = max(comparison, key=lambda name: comparison[name]["PR-AUC (AP)"])
    model, calibrator, threshold = fitted[winner]
    importance = permutation_importance(
        model, validation[FEATURES], validation.storm.astype(int),
        scoring="average_precision", n_repeats=5, random_state=SEED,
    )
    # Model, calibrator and operating threshold are frozen before scoring the test years.
    test = sets["Test"]
    test_scores = scores(test.storm, probability(model, calibrator, test[FEATURES]), threshold)
    usable = periods[FEATURES].notna().all(axis=1)
    periods = periods.copy()
    periods.loc[usable, "probability"] = probability(model, calibrator, periods.loc[usable, FEATURES])
    coverage = {
        name: {"Scored": len(sets[name]), "Total": int(periods.index.year.isin(range(start, end + 1)).sum())}
        for name, (start, end) in SPLITS.items()
    }
    return {
        "version": 1, "source": SOURCE, "features": FEATURES, "winner": winner,
        "model": model, "calibrator": calibrator, "threshold": threshold, "periods": periods,
        "validation": comparison, "test": test_scores, "coverage": coverage,
        "importance": dict(zip(FEATURES, importance.importances_mean.tolist())),
        "reference": training[FEATURES].median().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Read existing annual files only.")
    args = parser.parse_args()
    try:
        periods = load_periods(args.data_dir, download=not args.offline)
        if args.download_only:
            print(f"Verified {len(periods):,} three-hour bins in {args.data_dir}")
            return
        result = train(periods)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        joblib.dump(result, temporary, compress=3)
        temporary.replace(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"SolarShield could not finish. {error}\n"
                       f"Check the annual OMNI2 files in {args.data_dir}.\n")
    print(json.dumps({key: result[key] for key in ("winner", "validation", "test", "coverage")}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
