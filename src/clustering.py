from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from src.config import FEATURES,NUMERIC_FEATURES,SEED
from src.preprocessing import make_preprocessor
def fit_clustering(df):
 pre=make_preprocessor(); z=pre.fit_transform(df[FEATURES]); scores={}; models={}
 for k in range(2,7):
  m=KMeans(n_clusters=k,random_state=SEED,n_init=20); labels=m.fit_predict(z); scores[str(k)]=float(silhouette_score(z,labels)); models[k]=m
 k=int(max(scores,key=scores.get)); model=models[k]; pipe=Pipeline([("preprocess",pre),("model",model)]); labels=model.labels_; grouped=df.assign(cluster=labels).groupby("cluster")[NUMERIC_FEATURES].mean().round(3); sizes=df.assign(cluster=labels).groupby("cluster").size(); separation="weak separation" if scores[str(k)] < 0.25 else "moderate separation"; profiles={str(i):{"size":int(sizes.loc[i]),"means":{c:float(v) for c,v in grouped.loc[i].items()},"interpretation":f"Descriptive input profile; overall K={k} separation is {separation}. Cluster IDs have no inherent rank."} for i in grouped.index}; return pipe,scores,k,profiles,z,labels
