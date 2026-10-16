import hashlib
import numpy as np
import pandas as pd
from src.config import BASE_ROWS,DATA_PATH,ARTIFACTS,REGISTRATION_NUMBER,SEED,VERIFICATION_CODE,ensure_dirs
def generate_dataset(output_path=DATA_PATH,fingerprint_path=ARTIFACTS/"data_fingerprint.txt"):
    ensure_dirs(); rng=np.random.default_rng(SEED); n=BASE_ROWS
    rainfall=rng.uniform(600,1400,n); temp=rng.uniform(12,26,n); moisture=rng.uniform(20,80,n); fertilizer=rng.uniform(0,250,n); size=rng.uniform(.2,4,n); season=rng.choice(["A","B"],n); coop=rng.integers(0,2,n)
    y=(.004*rainfall+.06*moisture+.02*fertilizer+1.30*size-.25*np.abs(temp-18)+.70*coop+np.where(season=="A",.5,-.2)+rng.normal(0,.8,n)).clip(min=.5)
    df=pd.DataFrame({"rainfall_mm":rainfall,"avg_temp_c":temp,"soil_moisture_pct":moisture,"fertilizer_kg":fertilizer,"field_size_ha":size,"season":season,"cooperative_support":coop,"yield_tons":y})
    m=round(.05*n); df.loc[rng.choice(n,m,False),"soil_moisture_pct"]=np.nan; df.loc[rng.choice(n,m,False),"fertilizer_kg"]=np.nan; df=pd.concat([df,df.iloc[:3]],ignore_index=True)
    df["high_risk"]=(df.yield_tons/df.field_size_ha < (df.yield_tons/df.field_size_ha).median()).astype(int)
    df.to_csv(output_path,index=False,float_format="%.4f"); digest=hashlib.sha256(output_path.read_bytes()).hexdigest(); fingerprint_path.write_text(digest+"\n",encoding="utf8")
    print(f"Verification code: {VERIFICATION_CODE}\nRegistration number: {REGISTRATION_NUMBER}\nBase row count: {n}\nMissing-value counts:\n{df.isna().sum()}\nDuplicate count: {df.duplicated().sum()}\nFinal row count: {len(df)}\nDataset path: {output_path}\nSHA-256: {digest}"); return df
if __name__=="__main__": generate_dataset()
