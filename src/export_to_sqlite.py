"""
Phase 4a: Database Storage (SQLite + parameterised queries)
----------------------------------------------------------------
Exports the processed trip/delay data to a relational SQLite database,
demonstrating:
  - Relational schema design (Data Storage & Processing requirement)
  - Parameterised queries (Security requirement — prevents SQL injection)
  - Sample queries for the Database Export submission requirement

We convert from PySpark to Pandas only at this final export step (the
brief allows Pandas for "smaller aggregations... pre-visualisation
reshaping" — exporting to a single-machine relational DB is the same
category of operation). For a genuinely huge dataset, PySpark's JDBC
writer would be the production-grade approach; that alternative is
documented below and worth mentioning in your report's tool-justification
discussion.
"""

import os
import sqlite3
from pyspark.sql import SparkSession

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "bus_delay.db")

# We store a SAMPLE of the full dataset in SQLite (not all 5.9M rows) —
# SQLite is a single-file, single-writer database intended for
# lightweight/embedded use, not big-data-scale storage. Storing a
# representative sample here while keeping the full dataset in Parquet
# (already produced by earlier phases) is the correct division of labour,
# and is exactly the "memory-vs-distributed trade-off" the brief asks you
# to reflect on. Document this choice explicitly in your report.
SAMPLE_FRACTION_FOR_DB = 0.02  # ~118k rows from 5.9M — still well over
                                 # the 100k threshold on its own if needed


def build_spark_session():
    spark = (
        SparkSession.builder
        .appName("SQLite_Export")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def create_schema(conn):
    """Relational schema: routes / trips / delay_events, normalised to
    avoid repeating route metadata on every row."""
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS delay_events")
    cur.execute("DROP TABLE IF EXISTS trips")
    cur.execute("DROP TABLE IF EXISTS routes")

    cur.execute("""
        CREATE TABLE routes (
            route_id TEXT PRIMARY KEY,
            route_short_name TEXT,
            agency_id TEXT,
            route_type INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE trips (
            trip_id TEXT PRIMARY KEY,
            route_id TEXT,
            service_id TEXT,
            trip_headsign TEXT,
            direction_id INTEGER,
            FOREIGN KEY (route_id) REFERENCES routes(route_id)
        )
    """)

    cur.execute("""
        CREATE TABLE delay_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT,
            stop_id TEXT,
            stop_sequence INTEGER,
            arrival_time TEXT,
            delay_minutes REAL,
            is_disrupted INTEGER,
            disruption_reason TEXT,
            FOREIGN KEY (trip_id) REFERENCES trips(trip_id)
        )
    """)

    # Indexes for the query patterns a transport authority would actually run
    cur.execute("CREATE INDEX idx_delay_trip ON delay_events(trip_id)")
    cur.execute("CREATE INDEX idx_delay_stop ON delay_events(stop_id)")
    cur.execute("CREATE INDEX idx_trips_route ON trips(route_id)")

    conn.commit()
    print("[INFO] Schema created: routes, trips, delay_events")


def load_data_and_export(spark, conn):
    df = spark.read.parquet(os.path.join(PROCESSED_DIR, "trips_with_delay.parquet"))

    if SAMPLE_FRACTION_FOR_DB < 1.0:
        df = df.sample(fraction=SAMPLE_FRACTION_FOR_DB, seed=42)

    pdf = df.select(
        "route_id", "route_short_name", "agency_id", "route_type",
        "trip_id", "service_id", "trip_headsign", "direction_id",
        "stop_id", "stop_sequence", "arrival_time",
        "delay_minutes", "is_disrupted", "disruption_reason",
    ).toPandas()

    print(f"[INFO] Exporting {len(pdf):,} rows to SQLite")

    routes = pdf[["route_id", "route_short_name", "agency_id", "route_type"]].drop_duplicates(subset="route_id")
    trips = pdf[["trip_id", "route_id", "service_id", "trip_headsign", "direction_id"]].drop_duplicates(subset="trip_id")
    events = pdf[["trip_id", "stop_id", "stop_sequence", "arrival_time",
                   "delay_minutes", "is_disrupted", "disruption_reason"]]

    routes.to_sql("routes", conn, if_exists="append", index=False)
    trips.to_sql("trips", conn, if_exists="append", index=False)
    events.to_sql("delay_events", conn, if_exists="append", index=False)
    conn.commit()

    print(f"[INFO] Wrote {len(routes):,} routes, {len(trips):,} trips, "
          f"{len(events):,} delay events")


def run_sample_parameterised_queries(conn):
    """
    These are the queries to screenshot/include in your report's Appendix
    as evidence of parameterised query usage (Security requirement).
    Note the '?' placeholders — values are NEVER string-concatenated into
    the SQL, which is what prevents SQL injection.
    """
    cur = conn.cursor()

    print("\n=== Sample Query 1: Routes with average delay above a threshold ===")
    threshold_minutes = 5.0  # this would come from user input in a real app
    cur.execute("""
        SELECT r.route_short_name, AVG(e.delay_minutes) AS avg_delay, COUNT(*) AS n
        FROM delay_events e
        JOIN trips t ON e.trip_id = t.trip_id
        JOIN routes r ON t.route_id = r.route_id
        GROUP BY r.route_short_name
        HAVING AVG(e.delay_minutes) > ?
        ORDER BY avg_delay DESC
        LIMIT 10
    """, (threshold_minutes,))
    for row in cur.fetchall():
        print(row)

    print("\n=== Sample Query 2: Disruption events for a specific route (parameterised) ===")
    target_route = routes_sample_value(conn)
    cur.execute("""
        SELECT t.trip_id, e.stop_id, e.delay_minutes, e.disruption_reason
        FROM delay_events e
        JOIN trips t ON e.trip_id = t.trip_id
        WHERE t.route_id = ? AND e.is_disrupted = 1
        LIMIT 10
    """, (target_route,))
    for row in cur.fetchall():
        print(row)

    print("\n=== Sample Query 3: Overall on-time performance (parameterised bounds) ===")
    lower, upper = -1.0, 5.99  # DfT on-time definition
    cur.execute("""
        SELECT
            SUM(CASE WHEN delay_minutes BETWEEN ? AND ? THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
            AS pct_on_time
        FROM delay_events
    """, (lower, upper))
    print(cur.fetchone())


def routes_sample_value(conn):
    cur = conn.cursor()
    cur.execute("SELECT route_id FROM routes LIMIT 1")
    return cur.fetchone()[0]


def main():
    spark = build_spark_session()

    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh export each run

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    load_data_and_export(spark, conn)
    run_sample_parameterised_queries(conn)

    conn.close()
    spark.stop()

    print(f"\n[INFO] Database ready at {DB_PATH}")
    print("[ACTION] For your Database Export submission requirement:")
    print(f"  1. SQL dump: run  sqlite3 {DB_PATH} .dump > docs/schema_dump.sql")
    print("  2. Schema diagram: draw routes -> trips -> delay_events "
          "(1-to-many both) for your System Design section")


if __name__ == "__main__":
    main()
