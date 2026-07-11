"""
Phase 2b: Feature Engineering (PySpark)
------------------------------------------
Builds the feature set for the delay-prediction regression models and
writes a train/test split ready for Phase 3 (ml_models.py).

Features engineered:
  - hour_of_day, day_of_week_num (from arrival_time + calendar flags)
  - is_peak (7-9am / 4-6pm heuristic — common transport-analytics feature)
  - route_id_indexed, operator/service categorical encodings (StringIndexer)
  - stop_sequence (proxy for how far into the trip a stop is)
  - is_disrupted (also useful as a classification target if you want a
    secondary comparison, though the brief's chosen target here is
    delay_minutes for regression)

Target variable: delay_minutes
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DIR, "trips_with_delay.parquet")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.parquet")


def build_spark_session():
    spark = (
        SparkSession.builder
        .appName("Feature_Engineering_Bus_Delay")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def extract_time_features(df):
    """arrival_time in GTFS can exceed 24:00:00 (past-midnight trips), so we
    parse it manually rather than relying on a strict timestamp cast."""
    df = df.withColumn("arrival_hour", F.split("arrival_time", ":")[0].cast("int"))
    df = df.withColumn("arrival_hour_mod", F.col("arrival_hour") % 24)  # normalise past-midnight
    df = df.withColumn(
        "is_peak",
        F.when(
            (F.col("arrival_hour_mod").between(7, 9)) |
            (F.col("arrival_hour_mod").between(16, 18)),
            1
        ).otherwise(0)
    )
    # Day-of-week flags already exist in GTFS calendar.txt (monday..sunday = 0/1)
    day_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    available_day_cols = [c for c in day_cols if c in df.columns]
    if available_day_cols:
        day_expr = F.array(*[F.col(c).cast("int") for c in available_day_cols])
        df = df.withColumn("num_service_days", F.aggregate(day_expr, F.lit(0), lambda acc, x: acc + x))
    return df


def encode_categoricals(df):
    indexers = []
    for col in ["route_id", "agency_id"]:
        if col in df.columns:
            indexers.append(
                StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
            )
    if indexers:
        pipeline = Pipeline(stages=indexers)
        df = pipeline.fit(df).transform(df)
    return df


def select_feature_columns(df):
    candidate_features = [
        "arrival_hour_mod", "is_peak", "stop_sequence", "num_service_days",
        "route_id_idx", "agency_id_idx", "is_disrupted",
    ]
    features = [c for c in candidate_features if c in df.columns]
    print(f"[INFO] Using feature columns: {features}")
    return features


def main():
    spark = build_spark_session()

    df = spark.read.parquet(INPUT_PATH)
    df = extract_time_features(df)
    df = encode_categoricals(df)

    # Drop rows where the target is null (shouldn't happen, but defensive)
    df = df.filter(F.col("delay_minutes").isNotNull())

    # is_disrupted needs to be numeric for VectorAssembler
    if "is_disrupted" in df.columns:
        df = df.withColumn("is_disrupted", F.col("is_disrupted").cast("int"))

    feature_cols = select_feature_columns(df)

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )
    df = assembler.transform(df)

    model_df = df.select("features", F.col("delay_minutes").alias("label"))

    # 80/20 split, reproducible seed
    train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.repartition(8).cache()
    test_df = test_df.repartition(4).cache()

    print(f"[INFO] Train rows: {train_df.count():,}")
    print(f"[INFO] Test rows:  {test_df.count():,}")

    train_df.write.mode("overwrite").parquet(TRAIN_PATH)
    test_df.write.mode("overwrite").parquet(TEST_PATH)
    print(f"[INFO] Wrote {TRAIN_PATH} and {TEST_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()