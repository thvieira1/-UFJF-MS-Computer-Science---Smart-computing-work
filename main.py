import numpy as np
from fuzzy_anomaly_system import FuzzyAnomalyDetector


def compute_forecast_error(y_true, y_pred, eps=1e-6):
    """
    Compute normalized forecast error in [0, 1] using MAE divided by the data range.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    value_range = np.max(y_true) - np.min(y_true) + eps

    normalized = np.clip(mae / (value_range + eps), 0.0, 1.0)
    return float(normalized)


def compute_variance_change(window_data, baseline_data, eps=1e-6):
    """
    Compute normalized variance change in [0, 1].

    The score is based on the ratio between window variance and baseline variance,
    focusing on increases in variability.
    """
    window_data = np.asarray(window_data)
    baseline_data = np.asarray(baseline_data)

    var_window = np.var(window_data, axis=0)
    var_baseline = np.var(baseline_data, axis=0) + eps

    ratio = var_window / var_baseline
    ratio = np.clip(ratio, 0.0, 5.0)

    score = np.mean(ratio - 1.0)
    score = float(np.clip(score / 4.0, 0.0, 1.0))
    return score


def compute_correlation_change(window_data, baseline_data, eps=1e-6):
    """
    Compute normalized correlation change in [0, 1].

    The score is based on the Frobenius norm of the difference between
    correlation matrices of the window and the baseline.
    """
    window_data = np.asarray(window_data)
    baseline_data = np.asarray(baseline_data)

    corr_window = np.corrcoef(window_data, rowvar=False)
    corr_baseline = np.corrcoef(baseline_data, rowvar=False)

    diff = corr_window - corr_baseline
    frob = np.sqrt(np.sum(diff ** 2))

    d = corr_window.shape[0]
    max_possible = 2 * d
    score = float(np.clip(frob / (max_possible + eps), 0.0, 1.0))
    return score


def generate_synthetic_data(seed=42):
    """
    Generate synthetic multivariate time series data for:
        - baseline: normal regime
        - window_normal: normal window
        - window_anom: anomalous window
    """
    rng = np.random.default_rng(seed)

    baseline = rng.normal(0, 1, size=(500, 3))
    window_normal = rng.normal(0, 1, size=(100, 3))

    mean_shift = np.array([4.0, -4.0, 3.0])
    cov_anom = np.array(
        [
            [4.0, 3.0, 2.0],
            [3.0, 5.0, 2.5],
            [2.0, 2.5, 3.5],
        ]
    )
    window_anom = rng.multivariate_normal(mean_shift, cov_anom, size=100)

    return baseline, window_normal, window_anom



def compute_indicators_for_window(window, baseline):
    """
    Compute all three indicators (forecast error, variance change, correlation change)
    for a given window and baseline.
    """
    window = np.asarray(window)
    baseline = np.asarray(baseline)

    y_true = window[:, 0]
    y_pred = np.full_like(y_true, np.mean(baseline[:, 0]))

    ep = compute_forecast_error(y_true, y_pred)
    mv = compute_variance_change(window, baseline)
    mc = compute_correlation_change(window, baseline)

    return ep, mv, mc


def main():
    baseline, window_normal, window_anom = generate_synthetic_data()

    ep_norm, mv_norm, mc_norm = compute_indicators_for_window(window_normal, baseline)
    ep_anom, mv_anom, mc_anom = compute_indicators_for_window(window_anom, baseline)

    detector = FuzzyAnomalyDetector()

    crisp_normal, label_normal = detector.evaluate(ep_norm, mv_norm, mc_norm)
    crisp_anom, label_anom = detector.evaluate(ep_anom, mv_anom, mc_anom)

    print("=== Normal window indicators ===")
    print(f"Forecast error (EP):        {ep_norm:.3f}")
    print(f"Variance change (MV):       {mv_norm:.3f}")
    print(f"Correlation change (MC):    {mc_norm:.3f}")
    print(f"Anomaly level (crisp):      {crisp_normal:.3f}")
    print(f"Linguistic label:           {label_normal}")
    print()
    print("=== Anomalous window indicators ===")
    print(f"Forecast error (EP):        {ep_anom:.3f}")
    print(f"Variance change (MV):       {mv_anom:.3f}")
    print(f"Correlation change (MC):    {mc_anom:.3f}")
    print(f"Anomaly level (crisp):      {crisp_anom:.3f}")
    print(f"Linguistic label:           {label_anom}")


if __name__ == "__main__":
    main()
