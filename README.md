# Musanze Harvest Intelligence Service

Student: NZAYISENGA Erneste | Registration: 25/27341 | Verification: AI-A2-7341

This reproducible SWE 3513 prototype predicts synthetic potato harvest quantity, classifies high production risk, and groups farms with similar conditions. The institutional username is intentionally not invented and must be supplied by the student.

## Commands
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_all.py --student-id YOUR_INSTITUTIONAL_USERNAME --seed 7341
.venv\Scripts\python.exe predict.py --json samples\valid.json
.venv\Scripts\python.exe -m pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000
```

The generator uses `default_rng(7341)`, 242 base rows, deliberate missing values and three exact duplicates, and records SHA-256 `408c124768bd6ff6bb83987a961dff1077a34ee08873b70363626f95b57e5fa5`. A single stratified 80/20 split is shared by regression and classification. Model selection uses five-fold training-only cross-validation; the held-out test partition is used only for final metrics. K-Means evaluates K=2..6 using silhouette score. `/health`, `/model-info`, `/predict`, and `/docs` are provided by FastAPI.

The data are synthetic, not field observations. This is classroom decision support, not a replacement for an agronomist. Predictions require human review and must not determine access to finance, insurance, or public services. The student must personally verify the final report, institutional username, interpretations, and evidence.

GitHub repository and final commit hash: to be filled by the student.
