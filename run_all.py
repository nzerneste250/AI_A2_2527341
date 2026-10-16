import argparse,json,joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.generate_data import generate_dataset
from src.config import *
from src.validation import validate_identity,clean_data
from src.preprocessing import shared_split
from src.regression import select_and_evaluate as select_regression
from src.classification import select_and_evaluate as select_classification
from src.clustering import fit_clustering

def main(student_id,seed):
 validate_identity(seed,REGISTRATION_NUMBER,student_id); ensure_dirs(); df=clean_data(generate_dataset()); train_idx,test_idx=shared_split(df); train=df.loc[train_idx]; test=df.loc[test_idx]; Xtr=train[FEATURES]; Xte=test[FEATURES]
 reg,reg_name,reg_val,reg_final,_=select_regression(Xtr,train.yield_tons,Xte,test.yield_tons); clf,clf_name,clf_val,clf_final,_=select_classification(Xtr,train.high_risk,Xte,test.high_risk); joblib.dump(reg,MODELS/"regression.joblib"); joblib.dump(clf,MODELS/"classification.joblib")
 reg_art={"selection_method":"5-fold shuffled training-only cross-validation","validation_scores":reg_val,"selected_model":reg_name,"final_test_metrics":reg_final,"verification_code":VERIFICATION_CODE}; cls_art={"selection_method":"5-fold stratified training-only cross-validation","validation_scores":clf_val,"selected_model":clf_name,"final_test_metrics":clf_final,"confusion_matrix":clf_final[clf_name]["confusion_matrix"],"verification_code":VERIFICATION_CODE}; (ARTIFACTS/"regression_metrics.json").write_text(json.dumps(reg_art,indent=2)); (ARTIFACTS/"classification_metrics.json").write_text(json.dumps(cls_art,indent=2))
 rp=reg.predict(Xte); cp=clf.predict(Xte); prob=clf.predict_proba(Xte)[:,1]; plt.figure(); plt.scatter(test.yield_tons,rp-test.yield_tons); plt.axhline(0,color="red"); plt.savefig(PLOTS/"residuals.png"); plt.close(); plt.figure(); plt.imshow(clf_final[clf_name]["confusion_matrix"],cmap="Blues"); plt.colorbar(); plt.savefig(PLOTS/"confusion_matrix.png"); plt.close()
 cluster,scores,k,profiles,z,labels=fit_clustering(df); joblib.dump(cluster,MODELS/"clustering.joblib"); (ARTIFACTS/"clustering_report.json").write_text(json.dumps({"silhouette_scores":scores,"selected_k":k,"profiles":profiles,"interpretation_note":"Profiles are descriptive averages of permitted input features; they do not establish causal or quality rankings.","verification_code":VERIFICATION_CODE},indent=2)); plt.figure(); plt.scatter(z[:,0],z[:,1],c=labels); plt.savefig(PLOTS/"clusters.png"); plt.close()
 out=test.copy(); out.index.name="row_id"; out["predicted_yield_tons"]=rp; out["yield_error"]=rp-test.yield_tons; out["predicted_high_risk"]=cp; out["high_risk_probability"]=prob; out["cluster"]=cluster.predict(Xte); out.to_csv(ARTIFACTS/"test_predictions.csv")
 print(json.dumps({"verification_code":VERIFICATION_CODE,"fingerprint":(ARTIFACTS/"data_fingerprint.txt").read_text().strip(),"selected_regression":reg_name,"selected_classification":clf_name,"selected_cluster_k":k,"test_rows":len(test)},indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--student-id",required=True); p.add_argument("--seed",type=int,required=True); a=p.parse_args(); main(a.student_id,a.seed)
