"""
Phase 1: GTFS Ingestion (PySpark)
----------------------------------
Loads real BODS GTFS timetable data for one region, joins the core tables,
and demonstrates the distributed-computing requirements from the brief:
partitioning, caching, repartitioning, and Spark UI-visible stage execution.

Run this AFTER downloading a regional GTFS zip into data/raw/gtfs/
(see README.md Step 1).

While this runs, open http://localhost:4040 in a browser and screenshot the
"Stages" tab and "Storage" tab (for the cached DataFrame) — the brief
requires at least one Spark UI screenshot showing partition utilisation.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gtfs")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NUM_PARTITIONS = 8  # brief requires >= 4; we use 8 to allow room to justify the choice

os.makedirs(PROCESSED_DIR, exist_ok=True)


def build_spark_session():
    """
    SparkSession configuration is documented here (also copy this into your
    report's System Design section as evidence of configuration choices).
    """
    spark = (
        SparkSession.builder
        .appName("BODS_GTFS_Ingestion")
        .config("spark.sql.shuffle.partitions", str(NUM_PARTITIONS))
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Explicit schemas (faster + safer than inferSchema on large files)
STOP_TIMES_SCHEMA = StructType([
    StructField("trip_id", StringType(), True),
    StructField("arrival_time", StringType(), True),
    StructField("departure_time", StringType(), True),
    StructField("stop_id", StringType(), True),
    StructField("stop_sequence", IntegerType(), True),
    StructField("stop_headsign", StringType(), True),
    StructField("pickup_type", StringType(), True),
    StructField("drop_off_type", StringType(), True),
    StructField("shape_dist_traveled", DoubleType(), True),
    StructField("timepoint", StringType(), True),
])


def load_gtfs_tables(spark):
    stop_times = (
        spark.read.csv(
            os.path.join(RAW_DIR, "stop_times.txt"),
            header=True,
            schema=STOP_TIMES_SCHEMA,
        )
    )
    trips = spark.read.csv(os.path.join(RAW_DIR, "trips.txt"), header=True, inferSchema=True)
    routes = spark.read.csv(os.path.join(RAW_DIR, "routes.txt"), header=True, inferSchema=True)
    stops = spark.read.csv(os.path.join(RAW_DIR, "stops.txt"), header=True, inferSchema=True)
    calendar = spark.read.csv(os.path.join(RAW_DIR, "calendar.txt"), header=True, inferSchema=True)
    return stop_times, trips, routes, stops, calendar


def join_tables(stop_times, trips, routes, stops, calendar):
    """
    Joins stop_times -> trips -> routes -> stops -> calendar.
    This is the "Merging Strategy" you describe in the report (Section 4):
    linking by trip_id, route_id, stop_id, service_id.
    """
    df = (
        stop_times
        .join(trips, on="trip_id", how="inner")
        .join(routes, on="route_id", how="left")
        .join(stops, on="stop_id", how="left")
        .join(calendar, on="service_id", how="left")
    )
    return df


def main():
    spark = build_spark_session()

    stop_times, trips, routes, stops, calendar = load_gtfs_tables(spark)

    raw_count = stop_times.count()
    print(f"[INFO] Raw stop_times record count: {raw_count:,}")
    if raw_count < 100_000:
        print("[WARN] Below the 100,000-record threshold on its own — "
              "you'll need the synthetic augmentation step or a larger/"
              "multi-region download to meet the Big Data Scale Requirement.")

    joined = join_tables(stop_times, trips, routes, stops, calendar)

    # Repartition + cache: required distributed-computing demonstration
    joined = joined.repartition(NUM_PARTITIONS, "route_id")
    joined.cache()
    joined.count()  # materialise the cache (triggers a Spark job you can screenshot)

    print(f"[INFO] Joined DataFrame partitions: {joined.rdd.getNumPartitions()}")
    print(f"[INFO] Joined record count: {joined.count():,}")

    # Example aggregation using PySpark SQL functions (for your EDA section)
    joined.groupBy("route_short_name").agg(
        F.count("*").alias("num_stop_events"),
        F.countDistinct("trip_id").alias("num_trips"),
    ).orderBy(F.desc("num_stop_events")).show(10, truncate=False)

    # Persist processed output for the next phase
    out_path = os.path.join(PROCESSED_DIR, "joined_schedule.parquet")
    joined.write.mode("overwrite").parquet(out_path)
    print(f"[INFO] Wrote joined schedule to {out_path}")

    print("\n[ACTION] Open http://localhost:4040 now if the job hasn't exited, "
          "or check the Spark History Server, and screenshot the Stages + "
          "Storage tabs for your report.")

    spark.stop()


if __name__ == "__main__":
    main()
