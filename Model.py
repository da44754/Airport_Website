import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
pd.set_option("display.precision", 6)
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings("ignore")
df = pd.read_csv('Jan_2019_ontime.csv')
df.head()
df['DELAYED'] = (df['ARR_DEL15'].astype(bool) | df['DEP_DEL15'].astype(bool)).astype(int)
df.drop(['OP_CARRIER_AIRLINE_ID','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','Unnamed: 21','OP_CARRIER','ARR_DEL15','DEP_DEL15','CANCELLED', 'DIVERTED'], axis=1, inplace=True)
strings_columns =  list(df.dtypes[df.dtypes == 'object'].index)
numeric_columns = list(df.drop(strings_columns,axis=1))
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
def encode_categories(features):
    lb_make = preprocessing.LabelEncoder()
    for i in range(len(features)):
        df[features[i]] = lb_make.fit_transform(df[features[i]])
encode_categories(['OP_UNIQUE_CARRIER' , 'ORIGIN' , 'DEST' , 'DEP_TIME_BLK'])
X = df.drop('DELAYED',axis=1)
y = df['DELAYED']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)
print(y_train.shape)
rf = RandomForestClassifier(n_estimators=150, min_samples_split=5, max_features='sqrt', max_depth=20)
rf.fit(x_train,y_train) 
pickle.dump(rf,open('model.pkl','wb'))
