from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.config import FEATURES,NUMERIC_FEATURES,CATEGORICAL_FEATURES,SEED
def make_preprocessor(scale=True):
 n=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler() if scale else "passthrough")]); c=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]); return ColumnTransformer([("numeric",n,NUMERIC_FEATURES),("categorical",c,CATEGORICAL_FEATURES)])
def shared_split(df):
 tr,te=train_test_split(df.index,test_size=.2,random_state=SEED,stratify=df["high_risk"]); return tr.sort_values(),te.sort_values()
def split_data(df,target):
 tr,te=shared_split(df); return df.loc[tr,FEATURES],df.loc[te,FEATURES],df.loc[tr,target],df.loc[te,target]
