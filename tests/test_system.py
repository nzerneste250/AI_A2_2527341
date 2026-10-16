import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import app
from data.generate_data import generate_dataset
from src.config import DATA_PATH, FEATURES, VERIFICATION_CODE, REGISTRATION_NUMBER
from src.inference import predict_record
from src.validation import validate_identity, validate_raw, clean_data, assert_no_leakage
from src.preprocessing import shared_split
VALID={"rainfall_mm":1100,"avg_temp_c":19,"soil_moisture_pct":60,"fertilizer_kg":150,"field_size_ha":2.5,"season":"A","cooperative_support":1}
def test_valid_inference():
 r=predict_record(VALID); assert r["verification_code"]==VERIFICATION_CODE; assert 0<=r["high_risk_probability"]<=1
def test_generation_reproducibility():
    import tempfile
    with tempfile.TemporaryDirectory(dir="F:/AI_A2_2527341") as folder:
        from pathlib import Path
        root=Path(folder); a=root/"a.csv"; b=root/"b.csv"; generate_dataset(a,root/"a.sha"); generate_dataset(b,root/"b.sha"); assert a.read_bytes()==b.read_bytes()
def test_schema_and_quality():
 df=pd.read_csv(DATA_PATH); info=validate_raw(df); assert set(FEATURES+["yield_tons","high_risk"]).issubset(df.columns); assert info["duplicate_count"]==3; assert len(clean_data(df))==242
def test_invalid_seed_and_leakage():
 with pytest.raises(ValueError): validate_identity(1234,"25/27341")
 with pytest.raises(ValueError): validate_identity(7341,REGISTRATION_NUMBER,"wrong@example.com")
 with pytest.raises(ValueError): assert_no_leakage(FEATURES+["yield_tons"],"high_risk")

def test_shared_split_and_classification_artifact():
 df=pd.read_csv(DATA_PATH); train_a,test_a=shared_split(df); train_b,test_b=shared_split(df); assert train_a.equals(train_b); assert test_a.equals(test_b)
 artifact=json.load(open("artifacts/classification_metrics.json",encoding="utf8")); matrix=artifact["confusion_matrix"]; assert matrix==artifact["final_test_metrics"][artifact["selected_model"]]["confusion_matrix"]; assert len(matrix)==2 and all(len(row)==2 for row in matrix); assert sum(map(sum,matrix))==len(test_a)

def test_saved_models_and_prediction_file():
 result=predict_record(VALID); assert result["verification_code"]==VERIFICATION_CODE
 predictions=pd.read_csv("artifacts/test_predictions.csv"); assert len(predictions)==49; assert predictions["row_id"].is_unique; assert set(["yield_tons","high_risk","predicted_yield_tons","predicted_high_risk","high_risk_probability","cluster"]).issubset(predictions.columns)
def test_api_predict_and_model_info():
 c=TestClient(app); r=c.post("/predict",json=VALID); assert r.status_code==200; assert r.json()["verification_code"]==VERIFICATION_CODE; assert c.get("/model-info").status_code==200
def test_api_validation_errors():
 c=TestClient(app); assert c.post("/predict",json={**VALID,"rainfall_mm":9999}).status_code==422; assert c.post("/predict",json={**VALID,"season":"C"}).status_code==422; assert c.post("/predict",json={"rainfall_mm":1100}).status_code==422
