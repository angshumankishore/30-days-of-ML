import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import joblib


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
print(df.head())

print(df.info())
print(df.describe())

print((df == 0).sum())

col_with_zeros = ['Pregnancies','BloodPressure','SkinThickness','Insulin']

df[col_with_zeros] = df[col_with_zeros].replace(0,np.nan)

df.fillna(df.median(),inplace=True)
print(df.head())

X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_test,X_train,y_test,y_train =  train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_pred,y_test))
print("Precission: ",precision_score(y_pred,y_test))
print("Confusion Matrix : \n",confusion_matrix(y_pred,y_test))
print("\nclassification_report:\n ",classification_report(y_pred,y_test))

joblib.dump(model,'diabetes_model.pkl')
joblib.dump(scaler,'scaler.pkl')