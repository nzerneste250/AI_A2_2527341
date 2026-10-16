from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import predict_record, load_bundle
from src.config import VERIFICATION_CODE
class FarmInput(BaseModel):
    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4)
    season: str
    cooperative_support: int = Field(ge=0, le=1)
app = FastAPI(title="Musanze Harvest Intelligence Service")
@app.get("/health")
def health():
    try: load_bundle(); return {"status": "healthy"}
    except Exception as exc: raise HTTPException(503, str(exc))
@app.get("/model-info")
def model_info(): return {"verification_code": VERIFICATION_CODE, "models": ["regression", "classification", "clustering"]}
@app.post("/predict")
def predict(item: FarmInput):
    if item.season not in {"A", "B"}: raise HTTPException(422, "season must be A or B")
    try: return predict_record(item.model_dump())
    except Exception as exc: raise HTTPException(400, str(exc))
