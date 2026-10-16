import pandas as pd
from src.config import FEATURES,TARGETS,SEED
EXPECTED_STUDENT_ID="ug2527341@ines.ac.rw"
RANGES={"rainfall_mm":(600,1400),"avg_temp_c":(12,26),"soil_moisture_pct":(20,80),"fertilizer_kg":(0,250),"field_size_ha":(.2,4)}
def validate_identity(seed,registration_number,student_id=EXPECTED_STUDENT_ID):
    if student_id != EXPECTED_STUDENT_ID: raise ValueError(f"Unexpected institutional username; expected {EXPECTED_STUDENT_ID}")
    digits="".join(c for c in registration_number if c.isdigit())
    if len(digits)<4 or str(seed)!=digits[-4:] or seed!=SEED: raise ValueError("Seed must equal the last four registration-number digits and be 7341")
def validate_raw(df,seed=SEED):
    missing=[c for c in FEATURES+TARGETS if c not in df]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    for c,(lo,hi) in RANGES.items():
        v=df[c].dropna()
        if not pd.api.types.is_numeric_dtype(df[c]) or ((v<lo)|(v>hi)).any(): raise ValueError(f"Invalid values in {c}")
    if not set(df.season.dropna().unique()).issubset({"A","B"}): raise ValueError("season must be A or B")
    if not set(df.cooperative_support.dropna().unique()).issubset({0,1}): raise ValueError("cooperative_support must be 0 or 1")
    return {"missing_values":df[FEATURES+TARGETS].isna().sum().to_dict(),"duplicate_count":int(df.duplicated().sum()),"rows":len(df)}
def clean_data(df): validate_raw(df); return df.drop_duplicates().reset_index(drop=True)
def assert_no_leakage(columns,target):
    leaked=((set(TARGETS)|{"yield_per_ha"})&set(columns))-set([target])
    if leaked: raise ValueError(f"Target leakage detected: {sorted(leaked)}")
