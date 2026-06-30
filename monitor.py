"""
monitor.py
Phase 2: Drift monitoring.

Compares the statistical distribution of incoming ("current") transaction
data against the original training data ("reference") using the
Kolmogorov-Smirnov test per numeric feature, plus a population stability
index (PSI) for the categorical 'type' feature. If enough features drift
past a threshold, it flags the model as stale.

This is dependency-light by design (just scipy) so it's easy to run anywhere.
For a fuller dashboard, swap this logic into Evidently AI's Report objects.

Usage:
    python monitor.py --reference ../data/transactions.csv --current ../data/new_batch.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUMERIC_FEATURES = [
    "hour_of_day", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for a categorical feature. PSI > 0.2 generally indicates significant drift."""
    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)
    categories = set(ref_dist.index) | set(cur_dist.index)

    psi = 0.0
    for cat in categories:
        ref_pct = ref_dist.get(cat, 1e-6)
        cur_pct = cur_dist.get(cat, 1e-6)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi)


def check_drift(reference_path: str, current_path: str, ks_pvalue_threshold: float = 0.01,
                 drift_feature_fraction_threshold: float = 0.4) -> dict:
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    feature_results = {}
    drifted_count = 0

    for feature in NUMERIC_FEATURES:
        stat, p_value = ks_2samp(ref[feature], cur[feature])
        drifted = p_value < ks_pvalue_threshold
        drifted_count += int(drifted)
        feature_results[feature] = {"ks_statistic": round(float(stat), 4),
                                     "p_value": round(float(p_value), 6),
                                     "drifted": bool(drifted)}

    psi = population_stability_index(ref["type"], cur["type"])
    type_drifted = psi > 0.2
    drifted_count += int(type_drifted)
    feature_results["type"] = {"psi": round(psi, 4), "drifted": bool(type_drifted)}

    total_features = len(NUMERIC_FEATURES) + 1
    drift_fraction = drifted_count / total_features
    retrain_recommended = bool(drift_fraction >= drift_feature_fraction_threshold)

    result = {
        "drift_fraction": round(drift_fraction, 2),
        "drifted_feature_count": drifted_count,
        "total_features_checked": total_features,
        "retrain_recommended": retrain_recommended,
        "feature_details": feature_results,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="../data/transactions.csv")
    parser.add_argument("--current", type=str, default="../data/new_batch.csv")
    parser.add_argument("--out", type=str, default="../models/drift_report.json")
    args = parser.parse_args()

    report = check_drift(args.reference, args.current)
    print(json.dumps(report, indent=2))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    if report["retrain_recommended"]:
        print("\n⚠️  DRIFT DETECTED — retraining recommended. "
              "In the full pipeline this would trigger train_model.py automatically.")
    else:
        print("\n✅ No significant drift detected. Model is still healthy.")
