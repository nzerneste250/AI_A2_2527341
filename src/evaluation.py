import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
def regression_metrics(y_true,predictions): return {"mae":float(mean_absolute_error(y_true,predictions)),"rmse":float(np.sqrt(mean_squared_error(y_true,predictions))),"r2":float(r2_score(y_true,predictions))}
def classification_metrics(y_true,predictions,probabilities):
 r={"precision":float(precision_score(y_true,predictions,zero_division=0)),"recall":float(recall_score(y_true,predictions,zero_division=0)),"f1":float(f1_score(y_true,predictions,zero_division=0)),"confusion_matrix":confusion_matrix(y_true,predictions).tolist()}; r["roc_auc"]=float(roc_auc_score(y_true,probabilities)) if len(set(y_true))==2 else None; return r
