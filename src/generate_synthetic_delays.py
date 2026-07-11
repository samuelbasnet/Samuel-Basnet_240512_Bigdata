"""
Phase 1b: Synthetic Delay Layer (calibrated augmentation)
----------------------------------------------------------
The real GTFS schedule (loaded by ingest_gtfs.py) has no delay column —
DfT does not publicly archive historical AVL/real-time location data.
This script adds a synthetic `delay_minutes` target, calibrated to a real,
citable statistic:

    DfT Annual Bus Statistics (year ending March 2025): 80% of non-frequent
    bus services in England ran "on time", defined as between 1 minute
    early and 5 minutes 59 seconds late. Regional performance ranges
    76-85%. (gov.uk/government/statistics/annual-bus-statistics-year-ending-march-2025)

Methodology (put this in your report's Data Collection & Preprocessing
section, word-for-word or paraphrased — this IS your augmentation
justification):

  1. Each route is assigned an on-time probability drawn uniformly from
     [0.755, 0.845] — i.e. centred on the real 80% national figure, spread
     across the real 76-85% regional range reported by DfT. This gives
     route-to-route heterogeneity instead of one flat rate.
  2. For each stop-time event, a uniform random draw decides whether that
     event is "on time" (within the DfT-defined band) or a disruption.
  3. On-time events: delay ~ approx N(1.5, 1.3) minutes, clipped to
     [-1, 5.99] to respect the DfT definition exactly.
  4. Disrupted events: delay = 6 + Exponential(scale=8) minutes — a
     right-skewed tail typical of real-world congestion/breakdown delays,
     starting just above the on-time cutoff.
  5. A categorical disruption_reason is attached to disrupted events only,
     sampled from common real-world causes (traffic, breakdown, diversion,
     weather, staff shortage, signal fault) — this becomes your
     "Disruptions" catalogue for the multi-catalogue integration
     requirement.

This is implemented natively in PySpark (not pandas) so it scales to the
full joined dataset without collecting to the driver — this is itself worth
mentioning in your report's tool-justification discussion.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DIR, "joined_schedule.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "trips_with_delay.parquet")

DISRUPTION_REASONS = [
    "Traffic congestion",
    "Road closure",
    "Vehicle breakdown",
    "Staff shortage",
    "Weather conditions",
    "Diversion in operation",
    "Signal fault",
]


def build_spark_session():
    spark = (
        SparkSession.builder
        .appName("Synthetic_Delay_Augmentation")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def add_route_level_on_time_probability(df, seed=42):
    """Step 1: per-route on-time probability, calibrated to DfT 76-85% range."""
    routes = df.select("route_id").distinct()
    routes = routes.withColumn(
        "p_on_time",
        F.lit(0.755) + F.rand(seed) * F.lit(0.09)  # uniform in [0.755, 0.845]
    )
    return df.join(routes, on="route_id", how="left")


def add_synthetic_delay(df, seed=7):
    """Steps 2-4: simulate delay_minutes using Box-Muller for the normal
    component and inverse-CDF sampling for the exponential tail — both
    implemented with native Spark SQL functions so everything stays
    distributed."""

    df = df.withColumn("_u_disrupt", F.rand(seed))
    df = df.withColumn("is_disrupted", F.col("_u_disrupt") > F.col("p_on_time"))

    # Box-Muller normal(mean=1.5, std=1.3), clipped to DfT on-time band
    df = df.withColumn("_u1", F.rand(seed + 1))
    df = df.withColumn("_u2", F.rand(seed + 2))
    df = df.withColumn(
        "_z",
        F.sqrt(-2 * F.log("_u1")) * F.cos(F.lit(2 * 3.141592653589793) * F.col("_u2"))
    )
    df = df.withColumn("_ontime_delay", F.lit(1.5) + F.col("_z") * F.lit(1.3))
    df = df.withColumn(
        "_ontime_delay_clipped",
        F.greatest(F.lit(-1.0), F.least(F.lit(5.99), F.col("_ontime_delay")))
    )

    # Exponential tail for disruptions: 6 + Exp(scale=8)
    df = df.withColumn("_u3", F.rand(seed + 3))
    df = df.withColumn("_disrupted_delay", F.lit(6.0) + (-F.log("_u3") * F.lit(8.0)))

    df = df.withColumn(
        "delay_minutes",
        F.round(
            F.when(F.col("is_disrupted"), F.col("_disrupted_delay"))
             .otherwise(F.col("_ontime_delay_clipped")),
            2
        )
    )

    # Categorical disruption reason (Step 5), disrupted events only
    df = df.withColumn("_reason_idx", (F.rand(seed + 4) * len(DISRUPTION_REASONS)).cast("int"))
    reason_array = F.array(*[F.lit(r) for r in DISRUPTION_REASONS])
    df = df.withColumn(
        "disruption_reason",
        F.when(F.col("is_disrupted"), F.element_at(reason_array, F.col("_reason_idx") + 1))
         .otherwise(F.lit(None))
    )

    drop_cols = ["_u_disrupt", "_u1", "_u2", "_z", "_ontime_delay",
                 "_ontime_delay_clipped", "_u3", "_disrupted_delay", "_reason_idx"]
    return df.drop(*drop_cols)


def main():
    spark = build_spark_session()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"{INPUT_PATH} not found — run ingest_gtfs.py first to produce "
            "the joined real schedule data."
        )

    df = spark.read.parquet(INPUT_PATH)
    df = add_route_level_on_time_probability(df)
    df = add_synthetic_delay(df)
    df = df.repartition(8).cache()
    total = df.count()

    # Sanity check: does the simulated on-time rate land near 80%, per the
    # real DfT figure? Report this number in your report as validation
    # evidence for the augmentation.
    on_time = df.filter(~F.col("is_disrupted")).count()
    pct_on_time = 100 * on_time / total
    print(f"[INFO] Total records: {total:,}")
    print(f"[INFO] Simulated on-time rate: {pct_on_time:.1f}% "
          f"(target: ~80%, DfT national figure)")

    df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"[INFO] Wrote {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()