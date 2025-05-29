import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

df = pd.read_csv('heart-disease.csv')
#print(df.head())

#print(df.info())
#print(df.describe())

sns.countplot(x='target',data=df)
#plt.show()


#train test split 

X = df.drop('target',axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#creating pipeline 

pipe = Pipeline([
	('scaler',StandardScaler()),
	('model',LogisticRegression())
	])

#cross validation score 

cv_scores = cross_val_score(pipe,X_train,y_train,cv=5,scoring='accuracy')
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

pipe.fit(X_train,y_train)


y_pred = pipe.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
