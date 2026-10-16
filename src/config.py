from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED=7341
REGISTRATION_NUMBER="25/27341"
REGISTRATION_DIGITS="2527341"
VERIFICATION_CODE="AI-A2-7341"
BASE_ROWS=240+(SEED%41)
FEATURES=["rainfall_mm","avg_temp_c","soil_moisture_pct","fertilizer_kg","field_size_ha","season","cooperative_support"]
NUMERIC_FEATURES=FEATURES[:5]+[FEATURES[6]]
CATEGORICAL_FEATURES=["season"]
TARGETS=["yield_tons","high_risk"]
DATA_PATH=ROOT/"data"/"AI_A2_2527341.csv"
ARTIFACTS=ROOT/"artifacts"
MODELS=ROOT/"models"
PLOTS=ARTIFACTS/"plots"
def ensure_dirs():
    for p in [DATA_PATH.parent,ARTIFACTS,MODELS,PLOTS,ROOT/"samples",ROOT/"tests",ROOT/"docs",ROOT/"evidence"]: p.mkdir(parents=True,exist_ok=True)
