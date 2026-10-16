import argparse, json
from src.inference import predict_record
parser = argparse.ArgumentParser(); parser.add_argument("--json", required=True); args = parser.parse_args()
try: print(json.dumps(predict_record(json.loads(open(args.json, encoding="utf-8-sig").read())), indent=2))
except Exception as exc: print(json.dumps({"error": str(exc)}, indent=2)); raise SystemExit(1)
