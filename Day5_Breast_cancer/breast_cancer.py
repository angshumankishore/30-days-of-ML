import pandas as pd 
import joblib 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df.isnull().sum())

#scaling the data 
X = df.drop('target',axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train test spliting the data 	
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,random_state=42,test_size=0.2)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy: ",accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))

joblib.dump(model,"brest_cancer_model.pkl")
joblib.dump(scaler,"scaler.pkl")

