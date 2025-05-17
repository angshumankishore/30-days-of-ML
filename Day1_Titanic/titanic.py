import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("/home/angshuman/Projects/Ml_projects/30_days_ML/Day1_Titanic/train.csv") 
#print(df.head())

#getting basic info about the data 
#print(df.info())
#print(df.describe())

#plotting how many survived and how many didn't 
sns.countplot(x="Survived",data=df)
#plt.show()


#preporcessing data 
#dropping categorical data 
df = df.drop(columns=['Name','Cabin','Ticket'])
df['Age'] = df['Age'].fillna(df['Age'].mean())#filling age with mean values 
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) #filling with mode values 

df['Sex'] = df['Sex'].map({'male':0,'female':1}) #making categorical data to numerical values 
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2}) #making categorical data to numerical values 


#splitting the data 
X = df.drop('Survived',axis=1)
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#don't know why there are some na values present 
X_train = X_train.dropna()
y_train = y_train[X_train.index]
#making the model 

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#analysing the prediction 
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\n")
print("Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))
print("\n")
print("Classfication Report: ")
print(classification_report(y_test,y_pred))

