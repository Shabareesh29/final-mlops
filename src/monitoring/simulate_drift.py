from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD004.txt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "monitoring"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "simulated_drift_FD004.csv"
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
# LOAD TEST DATA
# =========================================================

print("Loading FD004 test data...")

data = pd.read_csv(
    TEST_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)

print(f"Original data shape: {data.shape}")


# =========================================================
# SIMULATE DATA DRIFT
# =========================================================

print("\nSimulating sensor drift...")

# Increase selected sensor values significantly.
data["sensor_2"] = data["sensor_2"] * 1.5
data["sensor_3"] = data["sensor_3"] * 1.5
data["sensor_4"] = data["sensor_4"] * 1.5


# =========================================================
# SAVE SIMULATED DATA
# =========================================================

data.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n====================================")
print("DRIFT DATA CREATED")
print("====================================")

print(f"Saved to:\n{OUTPUT_PATH}")
print(f"Shape: {data.shape}")