from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "artifact"
    / "production_model"
    / "random_forest_model.pkl"
)

SCALER_PATH = (
    BASE_DIR
    / "artifact"
    / "processed"
    / "scaler.pkl"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD004.txt"
)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Turbine Predictive Maintenance API",
    description="NASA CMAPSS FD004 RUL Prediction",
    version="1.0.0"
)


# =========================================================
# STATIC + TEMPLATES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

print("\nLoading production model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Production model not found: {MODEL_PATH}"
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✓ Production model loaded")
print("✓ Scaler loaded")


# =========================================================
# LOAD FD004 TEST DATA
# =========================================================

if not TEST_DATA_PATH.exists():

    raise FileNotFoundError(
        f"FD004 test dataset not found: "
        f"{TEST_DATA_PATH}"
    )


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


test_df = pd.read_csv(
    TEST_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)


print(
    f"✓ FD004 test data loaded: "
    f"{test_df.shape}"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class TurbineData(BaseModel):

    unit_number: int

    cycle: float

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


# =========================================================
# HOME PAGE
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "random_forest",
        "dataset": "NASA CMAPSS FD004"
    }


# =========================================================
# GET AVAILABLE TURBINES
# =========================================================

@app.get("/turbines")
def get_turbines():

    turbines = sorted(
        test_df["unit_number"]
        .unique()
        .tolist()
    )

    return {
        "turbines": turbines
    }


# =========================================================
# GET CYCLES FOR TURBINE
# =========================================================

@app.get("/turbines/{unit_number}/cycles")
def get_cycles(unit_number: int):

    turbine_data = test_df[
        test_df["unit_number"] == unit_number
    ]

    if turbine_data.empty:

        raise HTTPException(
            status_code=404,
            detail="Turbine not found."
        )

    cycles = sorted(
        turbine_data["cycle"]
        .astype(int)
        .tolist()
    )

    return {
        "unit_number": unit_number,
        "cycles": cycles
    }


# =========================================================
# GET SENSOR DATA
# =========================================================

@app.get(
    "/turbines/{unit_number}/cycles/{cycle}"
)
def get_turbine_data(
    unit_number: int,
    cycle: int
):

    row = test_df[
        (
            test_df["unit_number"]
            == unit_number
        )
        &
        (
            test_df["cycle"]
            == cycle
        )
    ]

    if row.empty:

        raise HTTPException(
            status_code=404,
            detail="Turbine/cycle combination not found."
        )

    result = row.iloc[0].to_dict()

    return result


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict_rul(data: TurbineData):

    try:

        input_data = data.model_dump()

        # -------------------------------------------------
        # Rename settings
        # -------------------------------------------------

        input_data["op_setting_1"] = (
            input_data.pop("setting_1")
        )

        input_data["op_setting_2"] = (
            input_data.pop("setting_2")
        )

        input_data["op_setting_3"] = (
            input_data.pop("setting_3")
        )

        # -------------------------------------------------
        # Features used by scaler
        # -------------------------------------------------

        feature_columns = [
            "op_setting_1",
            "op_setting_2",
            "op_setting_3"
        ] + [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        df = pd.DataFrame(
            [input_data]
        )

        # -------------------------------------------------
        # Scale 24 sensor/settings features
        # -------------------------------------------------

        scaled_features = scaler.transform(
            df[feature_columns]
        )

        scaled_features = pd.DataFrame(
            scaled_features,
            columns=feature_columns
        )

        # -------------------------------------------------
        # Final model input
        # cycle + 24 scaled features = 25
        # -------------------------------------------------

        model_input = pd.concat(
            [
                df[["cycle"]].reset_index(
                    drop=True
                ),

                scaled_features.reset_index(
                    drop=True
                )
            ],
            axis=1
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(
            model_input
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        # -------------------------------------------------
        # Health classification
        # -------------------------------------------------

        if prediction <= 30:

            status = "CRITICAL"

        elif prediction <= 60:

            status = "WARNING"

        else:

            status = "HEALTHY"

        return {
            "unit_number":
                data.unit_number,

            "cycle":
                data.cycle,

            "predicted_rul":
                round(prediction, 2),

            "status":
                status
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# INFO
# =========================================================

@app.get("/info")
def info():

    return {
        "application":
            "Turbine Predictive Maintenance",

        "dataset":
            "NASA CMAPSS FD004",

        "model":
            "Random Forest",

        "model_path":
            str(MODEL_PATH),

        "test_data":
            str(TEST_DATA_PATH)
    }