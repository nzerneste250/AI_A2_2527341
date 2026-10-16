from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_validate
from sklearn.pipeline import Pipeline
from src.config import SEED
from src.evaluation import regression_metrics
from src.preprocessing import make_preprocessor
def candidates(): return {"DummyRegressor":DummyRegressor(strategy="mean"),"LinearRegression":LinearRegression(),"RandomForestRegressor":RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=1)}
def select_and_evaluate(X_train,y_train,X_test,y_test):
 cv=KFold(n_splits=5,shuffle=True,random_state=SEED); validation={}; fitted={}
 for name,est in candidates().items():
  pipe=Pipeline([("preprocess",make_preprocessor()),("model",est)]); s=cross_validate(pipe,X_train,y_train,cv=cv,scoring={"rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"},n_jobs=1); validation[name]={"rmse_mean":float(-s["test_rmse"].mean()),"rmse_std":float(s["test_rmse"].std()),"mae_mean":float(-s["test_mae"].mean()),"r2_mean":float(s["test_r2"].mean())}; pipe.fit(X_train,y_train); fitted[name]=pipe
 selected=min(validation,key=lambda n:validation[n]["rmse_mean"]); final={n:regression_metrics(y_test,m.predict(X_test)) for n,m in fitted.items()}; return fitted[selected],selected,validation,final,fitted
