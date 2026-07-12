"""
Phase 3: Model Training & Comparison (PySpark MLlib)
-------------------------------------------------------
Trains and compares 3 regression models to predict delay_minutes:
  1. Linear Regression      - baseline, O(n*d) per iteration (L-BFGS)
  2. Random Forest Regressor - parallel ensemble, O(n log n * d * numTrees)
  3. GBT Regressor           - sequential boosted ensemble, same per-tree
                                cost as RF but trees built sequentially,
                                so wall-clock scales roughly linearly with
                                numTrees (no parallelism across trees)

Evaluated with RMSE, MAE, R² (regression metrics required by the brief),
plus wall-clock training time -> feeds the brief's "Model Efficiency"
metric (accuracy per second of training).

Uses train.parquet / test.parquet produced by feature_engineering.py.
"""

import os
import json
import time
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.parquet")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
FIG_DIR = os.path.join(DOCS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# On a laptop, 4.6M rows is genuinely heavy for RF/GBT (both build many
# trees over the full dataset). We train on a documented, reproducible
# sample of the training set to keep runtimes reasonable, while evaluating
# on the FULL test set so the reported metrics remain honest. This
# efficiency-vs-accuracy tradeoff is itself worth a paragraph in your
# Critical Reflection section (memory-vs-distributed / compute-cost
# tradeoff the brief explicitly asks about).
TRAIN_SAMPLE_FRACTION = 0.25  # set to 1.0 if your machine can handle full data
RANDOM_SEED = 42


def build_spark_session():
    spark = (
        SparkSession.builder
        .appName("ML_Model_Comparison_Bus_Delay")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def evaluate(predictions, model_name, train_seconds):
    results = {}
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
        results[metric] = evaluator.evaluate(predictions)

    # Brief's "Model Efficiency" metric: accuracy achieved per second of training.
    # We use 1/RMSE as an "accuracy-like" score since lower RMSE = better.
    results["train_seconds"] = train_seconds
    results["efficiency_score_per_sec"] = round((1 / results["rmse"]) / train_seconds, 6) if train_seconds > 0 else None
    results["model"] = model_name

    print(f"\n=== {model_name} ===")
    print(f"RMSE: {results['rmse']:.3f} min | MAE: {results['mae']:.3f} min | "
          f"R²: {results['r2']:.3f}")
    print(f"Training time: {train_seconds:.1f}s | "
          f"Efficiency score: {results['efficiency_score_per_sec']}")
    return results


def main():
    spark = build_spark_session()

    train_df = spark.read.parquet(TRAIN_PATH)
    test_df = spark.read.parquet(TEST_PATH)

    if TRAIN_SAMPLE_FRACTION < 1.0:
        train_df = train_df.sample(fraction=TRAIN_SAMPLE_FRACTION, seed=RANDOM_SEED)
        print(f"[INFO] Training on a {TRAIN_SAMPLE_FRACTION*100:.0f}% sample "
              f"of the training set (documented efficiency tradeoff).")

    train_df = train_df.cache()
    test_df = test_df.cache()
    print(f"[INFO] Train rows used: {train_df.count():,}")
    print(f"[INFO] Test rows used:  {test_df.count():,}")

    all_results = []

    # --- 1. Linear Regression (baseline) ---
    lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50)
    t0 = time.time()
    lr_model = lr.fit(train_df)
    t_lr = time.time() - t0
    lr_preds = lr_model.transform(test_df)
    all_results.append(evaluate(lr_preds, "Linear Regression", t_lr))

    # --- 2. Random Forest Regressor ---
    # maxBins must be >= the number of categories in the largest categorical
    # feature. route_id_idx has ~1,243 distinct routes (regional GTFS), so
    # the Spark default of 32 fails outright. 1300 covers it with headroom;
    # this does increase split-finding cost per node, which is worth noting
    # in your report's Algorithmic Complexity section.
    rf = RandomForestRegressor(featuresCol="features", labelCol="label",
                                numTrees=50, maxDepth=8, maxBins=1300, seed=RANDOM_SEED)
    t0 = time.time()
    rf_model = rf.fit(train_df)
    t_rf = time.time() - t0
    rf_preds = rf_model.transform(test_df)
    all_results.append(evaluate(rf_preds, "Random Forest", t_rf))

    # --- 3. GBT Regressor ---
    gbt = GBTRegressor(featuresCol="features", labelCol="label",
                        maxIter=50, maxDepth=5, maxBins=1300, seed=RANDOM_SEED)
    t0 = time.time()
    gbt_model = gbt.fit(train_df)
    t_gbt = time.time() - t0
    gbt_preds = gbt_model.transform(test_df)
    all_results.append(evaluate(gbt_preds, "Gradient Boosted Trees", t_gbt))

    # --- Feature importance (RF/GBT only — useful for the report) ---
    print("\n[INFO] Random Forest feature importances:", rf_model.featureImportances)
    print("[INFO] GBT feature importances:", gbt_model.featureImportances)

    # --- Save results table for the report ---
    results_df = pd.DataFrame(all_results)[
        ["model", "rmse", "mae", "r2", "train_seconds", "efficiency_score_per_sec"]
    ]
    results_path = os.path.join(DOCS_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved comparison table to {results_path}")
    print(results_df.to_string(index=False))

    # --- Comparison chart ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="rmse", palette="viridis")
    plt.title("Model Comparison: RMSE (lower is better)")
    plt.ylabel("RMSE (minutes)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_rmse_comparison.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="r2", palette="magma")
    plt.title("Model Comparison: R² (higher is better)")
    plt.ylabel("R²")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_r2_comparison.png"), dpi=150)
    plt.close()

    print(f"[INFO] Saved comparison charts to {FIG_DIR}")

    # Pick best model by RMSE for the report's headline result
    best = results_df.loc[results_df["rmse"].idxmin()]
    print(f"\n[RESULT] Best model by RMSE: {best['model']} "
          f"(RMSE={best['rmse']:.3f}, R²={best['r2']:.3f})")

    spark.stop()


if __name__ == "__main__":
    main()