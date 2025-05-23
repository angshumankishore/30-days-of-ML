import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle 

df = pd.read_csv("winequality-red.csv",sep=";")
print(df.head())


#basic EDA 

#checking the null values 

#print(df.isnull().sum()) #sums the null values for the all the columns 

#print(df.info())
#print(df.describe())

sns.countplot(x='quality',data=df)
plt.title("Distribution of Wine Quality Ranges")
#plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation heatmap of features ")
#plt.show()


#preparing the data to be processed 
X = df.drop("quality",axis=1)
y = df["quality"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

model = RandomForestClassifier(n_estimators=1000,random_state=42)
model.fit(X_train,y_train)

#evaluating the model 

y_pred = model.predict(X_test)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features ")
plt.show()


with open("wine-quality-model.pkl","wb") as f:
	pickle.dump((model,X.columns.tolist()),f)