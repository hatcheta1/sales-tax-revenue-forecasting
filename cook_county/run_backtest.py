"""
run_backtest.py
================
Evaluates all (target-formulation x model) combinations via rolling-
origin backtest and saves the comparison tables. This is the "does the
de-leveling approach work, and which combo is best" script.

Reframing implemented (see chmr_model_lib.py for the functions):
  1. Model de-leveled targets instead of raw CHMR level:
       trend_dev   : log(CHMR) minus a fitted log-linear growth trend
       mom_logdiff : month-over-month log-change
       yoy_logdiff : year-over-year log-change
  2. Add calendar month dummies (Feb..Dec, Jan baseline) to every model.
  3. Reconstruct absolute-dollar CHMR from each de-leveled prediction and
     evaluate ONLY on reconstructed levels (R^2, MAPE, RMSE), against
     benchmarks R^2 > 0.93, MAPE < 3.5%, OOS RMSE < $8M.
  4. Rolling-origin (walk-forward) backtest over the last N_TEST months
     for Ridge, RandomForest, and GradientBoosting.

Run:  python3 run_backtest.py
Outputs (written next to this script):
  chmr_delevel_results.csv           - metrics for all 9 combos
  chmr_month_dummy_effects.csv       - seasonal coefficients vs. January
  chmr_best_backtest_predictions.csv - actual vs. predicted for the winner
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from chmr_model_lib import (
    prepare_full_dataset, feature_cols_for, backtest, N_TEST,
)

RESULTS_DIR = Path("cook_county/results")

if __name__ == "__main__":
    df = prepare_full_dataset()

    results = {}
    detail = {}
    for target_kind in ["trend_dev", "mom_logdiff", "yoy_logdiff"]:
        for model_name in ["Ridge", "RandomForest", "GradientBoosting"]:
            key = f"{target_kind} | {model_name}"
            try:
                metrics, det = backtest(df, target_kind, model_name)
                results[key] = metrics
                detail[key] = det
            except Exception as e:
                results[key] = {"error": str(e)}

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("MAPE_%")
    print("\n=== Rolling-origin backtest (last {} months), reconstructed levels ===\n".format(N_TEST))
    print(results_df.round(3).to_string())

    print("\n--- vs. benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M) ---")
    for key, row in results_df.iterrows():
        if "error" in row and not pd.isna(row.get("error", np.nan)):
            continue
        passed = (row["R2"] > 0.93) and (row["MAPE_%"] < 3.5) and (row["RMSE"] < 8_000_000)
        print(f"{key:35s} {'PASS' if passed else 'fail'}")

    results_df.to_csv(RESULTS_DIR / "chmr_delevel_results.csv")

    # ---- month-dummy coefficients from a full-sample OLS on trend_dev ----
    feats = feature_cols_for("trend_dev", df.columns)
    fit_df = df.dropna(subset=feats + ["trend_dev"])
    X = sm.add_constant(fit_df[feats].astype(float))
    ols = sm.OLS(fit_df["trend_dev"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    month_coefs = ols.params[[c for c in feats if c.startswith("m_")]]
    month_pvals = ols.pvalues[[c for c in feats if c.startswith("m_")]]
    month_table = pd.DataFrame({"coef_log_dev": month_coefs, "p_value": month_pvals})
    month_table["approx_pct_effect_vs_Jan"] = (np.exp(month_table["coef_log_dev"]) - 1) * 100
    month_table.to_csv(RESULTS_DIR / "chmr_month_dummy_effects.csv")
    print("\n=== Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===\n")
    print(month_table.round(4).to_string())

    best_key = results_df.index[0]
    detail[best_key].to_csv(RESULTS_DIR / "chmr_best_backtest_predictions.csv", index=False)
    print(f"\nBest performing combo: {best_key}")

# === Rolling-origin backtest (last 24 months), reconstructed levels ===

#                                    R2  MAPE_%          RMSE
# mom_logdiff | RandomForest      0.779   3.786  4.277207e+06
# yoy_logdiff | GradientBoosting  0.640   4.510  5.451004e+06
# yoy_logdiff | RandomForest      0.622   4.663  5.585583e+06
# mom_logdiff | Ridge             0.636   4.713  5.483943e+06
# trend_dev | GradientBoosting    0.588   4.858  5.835469e+06
# trend_dev | RandomForest        0.145   6.813  8.402532e+06
# mom_logdiff | GradientBoosting -0.069   6.848  9.396990e+06
# yoy_logdiff | Ridge            -0.272   7.781  1.024984e+07
# trend_dev | Ridge              -0.631  10.157  1.160702e+07

# --- vs. benchmark thresholds (R2>0.93, MAPE<3.5%, OOS RMSE<$8M) ---
# mom_logdiff | RandomForest          fail
# yoy_logdiff | GradientBoosting      fail
# yoy_logdiff | RandomForest          fail
# mom_logdiff | Ridge                 fail
# trend_dev | GradientBoosting        fail
# trend_dev | RandomForest            fail
# mom_logdiff | GradientBoosting      fail
# yoy_logdiff | Ridge                 fail
# trend_dev | Ridge                   fail

# === Month effects (relative to January baseline), from HAC-robust OLS on trend deviation ===

#       coef_log_dev  p_value  approx_pct_effect_vs_Jan
# m_2         0.0847   0.2542                    8.8434
# m_3         0.3081   0.0006                   36.0843
# m_4        -0.0051   0.9634                   -0.5130
# m_5        -0.0183   0.8975                   -1.8118
# m_6         0.0845   0.7279                    8.8175
# m_7        -0.2320   0.5335                  -20.7091
# m_8        -0.2376   0.5616                  -21.1495
# m_9        -0.0516   0.8600                   -5.0283
# m_10        0.0657   0.6997                    6.7943
# m_11        0.0274   0.8201                    2.7784
# m_12       -0.0317   0.7070                   -3.1197

# Best performing combo: mom_logdiff | RandomForest