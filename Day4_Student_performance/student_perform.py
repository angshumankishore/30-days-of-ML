import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("student-por.csv",sep=';')
print(df.head())

label_encoder = {}

#encoding  the categorical features 
for col in df.select_dtypes(include='object').columns: 
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])
	label_encoder[col] = le

X = df.drop(['G3'],axis=1)
y = df['G3']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state = 42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("R2 score:  ", r2_score(y_test,y_pred))
print("RMSE:  ", np.sqrt(mean_squared_error(y_test,y_pred)))
print("MAE:  ", mean_squared_error(y_test,y_pred))


joblib.dump(model,"student-por.pkl")


