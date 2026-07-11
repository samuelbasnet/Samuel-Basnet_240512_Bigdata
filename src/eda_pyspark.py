"""
Phase 2a: Exploratory Data Analysis (PySpark)
-----------------------------------------------
Profiles trips_with_delay.parquet (output of generate_synthetic_delays.py):
  - null counts + cardinality per column (data quality assessment)
  - describe() + groupBy/agg statistical measures (mean, median, std,
    skewness, kurtosis) using native PySpark functions
  - IQR-based outlier detection on delay_minutes
  - visualisations: convert to Pandas ONLY at the final plotting step
    (as required by the brief), save PNGs to docs/figures/

Run after generate_synthetic_delays.py has produced
data/processed/trips_with_delay.parquet.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DIR, "trips_with_delay.parquet")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")


def build_spark_session():
    spark = (
        SparkSession.builder
        .appName("EDA_Bus_Delay")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def null_and_cardinality_report(df):
    print("\n=== Data Quality: Null Counts & Cardinality ===")
    total = df.count()
    rows = []
    for c in df.columns:
        null_count = df.filter(F.col(c).isNull()).count()
        distinct_count = df.select(c).distinct().count()
        rows.append((c, null_count, round(100 * null_count / total, 2), distinct_count))
    report = spark_session_global.createDataFrame(
        rows, ["column", "null_count", "null_pct", "distinct_count"]
    )
    report.orderBy(F.desc("null_pct")).show(50, truncate=False)
    return report


def statistical_summary(df):
    print("\n=== Statistical Summary: delay_minutes ===")
    df.select("delay_minutes").describe().show()

    stats = df.select(
        F.mean("delay_minutes").alias("mean"),
        F.expr("percentile_approx(delay_minutes, 0.5)").alias("median"),
        F.stddev("delay_minutes").alias("stddev"),
        F.skewness("delay_minutes").alias("skewness"),
        F.kurtosis("delay_minutes").alias("kurtosis"),
        F.min("delay_minutes").alias("min"),
        F.max("delay_minutes").alias("max"),
    ).collect()[0]

    print(f"Mean: {stats['mean']:.2f} | Median: {stats['median']:.2f} | "
          f"Std: {stats['stddev']:.2f}")
    print(f"Skewness: {stats['skewness']:.2f} | Kurtosis: {stats['kurtosis']:.2f}")
    print(f"Min: {stats['min']:.2f} | Max: {stats['max']:.2f}")
    print("[NOTE] Positive skew is expected: most trips run close to "
          "schedule, with a long right tail of disrupted trips. Report "
          "this in your EDA section as evidence the synthetic distribution "
          "behaves like real-world delay data.")
    return stats


def outlier_detection_iqr(df):
    print("\n=== Outlier Detection (IQR method) on delay_minutes ===")
    q1, q3 = df.approxQuantile("delay_minutes", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df.filter((F.col("delay_minutes") < lower) | (F.col("delay_minutes") > upper))
    n_outliers = outliers.count()
    total = df.count()
    print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"Outlier bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"Outliers: {n_outliers:,} ({100*n_outliers/total:.2f}% of records)")
    print("[NOTE] These 'outliers' are mostly your synthetic disruption "
          "tail (Exponential component) — expected, not data errors. "
          "State this explicitly in your report so it doesn't read as an "
          "unaddressed data quality issue.")
    return lower, upper, n_outliers


def route_level_aggregation(df):
    print("\n=== Route-level aggregation (groupBy/agg) ===")
    agg = (
        df.groupBy("route_short_name")
        .agg(
            F.count("*").alias("num_records"),
            F.mean("delay_minutes").alias("avg_delay"),
            F.stddev("delay_minutes").alias("stddev_delay"),
            F.sum(F.col("is_disrupted").cast("int")).alias("num_disrupted"),
        )
        .orderBy(F.desc("avg_delay"))
    )
    agg.show(15, truncate=False)
    return agg


def make_visualisations(df):
    """Convert to Pandas ONLY here, at the final plotting step, per brief."""
    print("\n=== Generating visualisations (Pandas conversion at final step only) ===")

    # Sample for plotting — never collect the full big-data set to the driver
    sample_pdf = df.select("delay_minutes", "is_disrupted", "route_short_name") \
                    .sample(fraction=0.05, seed=42).toPandas()

    # 1. Delay distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(sample_pdf["delay_minutes"], bins=50, kde=True)
    plt.axvline(5.99, color="red", linestyle="--", label="DfT on-time cutoff (5:59)")
    plt.title("Distribution of Delay (minutes)")
    plt.xlabel("Delay (minutes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delay_distribution.png"), dpi=150)
    plt.close()

    # 2. On-time vs disrupted proportion
    plt.figure(figsize=(6, 5))
    sample_pdf["is_disrupted"].value_counts(normalize=True).plot(
        kind="bar", color=["#4C72B0", "#DD8452"]
    )
    plt.title("Proportion On-Time vs Disrupted (sample)")
    plt.xticks([0, 1], ["On-time", "Disrupted"], rotation=0)
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "on_time_vs_disrupted.png"), dpi=150)
    plt.close()

    # 3. Top 10 routes by average delay
    top_routes_pdf = (
        df.groupBy("route_short_name")
        .agg(F.mean("delay_minutes").alias("avg_delay"), F.count("*").alias("n"))
        .filter(F.col("n") > 100)  # avoid noisy tiny-sample routes
        .orderBy(F.desc("avg_delay"))
        .limit(10)
        .toPandas()
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_routes_pdf, x="avg_delay", y="route_short_name", orient="h")
    plt.title("Top 10 Routes by Average Delay")
    plt.xlabel("Average delay (minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "top_routes_by_delay.png"), dpi=150)
    plt.close()

    print(f"[INFO] Saved 3 figures to {FIG_DIR}")


def main():
    global spark_session_global
    spark = build_spark_session()
    spark_session_global = spark

    df = spark.read.parquet(INPUT_PATH)
    df.cache()

    null_and_cardinality_report(df)
    statistical_summary(df)
    outlier_detection_iqr(df)
    route_level_aggregation(df)
    make_visualisations(df)

    spark.stop()


if __name__ == "__main__":
    main()