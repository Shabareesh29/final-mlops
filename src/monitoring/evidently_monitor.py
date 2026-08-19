from pathlib import Path
import json

import pandas as pd
import yaml

from evidently import Report
from evidently.presets import DataDriftPreset


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "monitoring_config.yaml"
)

REFERENCE_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "train_FD004.txt"
)

CURRENT_DATA_PATH = (
    BASE_DIR
    / "data"
    / "monitoring"
    / "simulated_drift_FD004.csv"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "evidently"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# OUTPUT FILES
# =========================================================

HTML_REPORT_PATH = (
    REPORT_DIR
    / "fd004_drift_report.html"
)

JSON_REPORT_PATH = (
    REPORT_DIR
    / "fd004_drift_results.json"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "monitoring_summary.json"
)


# =========================================================
# LOAD CONFIGURATION
# =========================================================

print("=" * 55)
print("EVIDENTLY DATA DRIFT MONITORING")
print("=" * 55)

print("\nLoading monitoring configuration...")

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = yaml.safe_load(file)


monitoring_config = config["monitoring"]

WARNING_THRESHOLD = monitoring_config[
    "drift_warning_threshold"
]

CRITICAL_THRESHOLD = monitoring_config[
    "drift_critical_threshold"
]


print(
    f"Warning threshold: "
    f"{WARNING_THRESHOLD * 100}%"
)

print(
    f"Critical threshold: "
    f"{CRITICAL_THRESHOLD * 100}%"
)


# =========================================================
# FD004 COLUMN NAMES
# =========================================================

COLUMN_NAMES = (
    ["unit_number", "cycle"]
    + [
        f"setting_{i}"
        for i in range(1, 4)
    ]
    + [
        f"sensor_{i}"
        for i in range(1, 22)
    ]
)


# =========================================================
# LOAD REFERENCE DATA
# =========================================================

print("\nLoading reference FD004 training data...")

reference_data = pd.read_csv(
    REFERENCE_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)


# =========================================================
# LOAD CURRENT DATA
# =========================================================

print("Loading current production data...")

current_data = pd.read_csv(
    CURRENT_DATA_PATH
)


print(
    f"\nReference data shape: "
    f"{reference_data.shape}"
)

print(
    f"Current data shape: "
    f"{current_data.shape}"
)


# =========================================================
# REMOVE IDENTIFIERS
# =========================================================

reference_data = reference_data.drop(
    columns=[
        "unit_number",
        "cycle"
    ]
)

current_data = current_data.drop(
    columns=[
        "unit_number",
        "cycle"
    ]
)


# =========================================================
# RUN EVIDENTLY
# =========================================================

print("\nRunning Evidently drift analysis...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data
)


# =========================================================
# SAVE HTML REPORT
# =========================================================

snapshot.save_html(
    str(HTML_REPORT_PATH)
)


# =========================================================
# GET EVIDENTLY RESULTS
# =========================================================

evidently_json = snapshot.json()

evidently_results = json.loads(
    evidently_json
)


# =========================================================
# SAVE RAW JSON
# =========================================================

if monitoring_config["save_json"]:

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(evidently_json)


# =========================================================
# FIND DRIFT RESULTS
# =========================================================

drifted_columns = 0
total_columns = len(current_data.columns)

for metric in evidently_results.get(
    "metrics",
    []
):

    metric_id = metric.get(
        "metric_id",
        ""
    )

    value = metric.get(
        "value"
    )

    if isinstance(value, dict):

        if (
            "count" in value
            and "share" in value
        ):

            drifted_columns = value["count"]

            break


# =========================================================
# CALCULATE DRIFT SHARE
# =========================================================

if total_columns > 0:

    drift_share = (
        drifted_columns
        / total_columns
    )

else:

    drift_share = 0


# =========================================================
# DETERMINE GOVERNANCE STATUS
# =========================================================

if drift_share >= CRITICAL_THRESHOLD:

    governance_status = "CRITICAL"

elif drift_share >= WARNING_THRESHOLD:

    governance_status = "WARNING"

else:

    governance_status = "HEALTHY"


# =========================================================
# CREATE MONITORING SUMMARY
# =========================================================

summary = {

    "project":
        "turbine-predictive-maintenance",

    "dataset":
        "NASA C-MAPSS FD004",

    "monitor":
        "data_drift",

    "reference_dataset":
        str(REFERENCE_DATA_PATH),

    "current_dataset":
        str(CURRENT_DATA_PATH),

    "total_columns":
        total_columns,

    "drifted_columns":
        drifted_columns,

    "drift_share":
        drift_share,

    "drift_share_percentage":
        drift_share * 100,

    "warning_threshold":
        WARNING_THRESHOLD,

    "critical_threshold":
        CRITICAL_THRESHOLD,

    "governance_status":
        governance_status,

    "evidently_version":
        "0.7.21",

    "status":
        "completed"
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 55)
print("EVIDENTLY MONITORING COMPLETED")
print("=" * 55)

print(
    f"\nTotal columns: "
    f"{total_columns}"
)

print(
    f"Drifted columns: "
    f"{drifted_columns}"
)

print(
    f"Drift share: "
    f"{drift_share * 100:.2f}%"
)

print(
    f"Governance status: "
    f"{governance_status}"
)

print(
    f"\nHTML report:"
    f"\n{HTML_REPORT_PATH}"
)

print(
    f"\nJSON results:"
    f"\n{JSON_REPORT_PATH}"
)

print(
    f"\nMonitoring summary:"
    f"\n{SUMMARY_PATH}"
)

print("\nStatus: SUCCESS")