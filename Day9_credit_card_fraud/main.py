import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
import joblib
import kagglehub
import os


path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path,"creditcard.csv")
df  = pd.read_csv(csv_path)
#print(df.head())


#counts the number of actual value of class types 
#print(df['Class'].value_counts())

#sns.countplot(x='Class',data=df)
#plt.show()

#plt.figure(figsize=(12,9))
#sns.heatmap(df.corr(),cmap='coolwarm',vmax=0.8)
#plt.title("Correlation Heatmap")
#plt.show()


X = df.drop('Class',axis=1)
y = df['Class']

rus = RandomUnderSampler(random_state=42)
X_resampled,y_resampled = rus.fit_resample(X,y)

X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled,y_train)	
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test,y_pred))
print("ROC AUC score : ",roc_auc_score(y_test,model.predict_proba(X_test_scaled)[:,1]))


fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()