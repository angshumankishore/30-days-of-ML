import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


boston = fetch_openml(name='boston',version=1,as_frame=True)
df = boston.frame 

#no null values exist in the dataset 

df.isnull().sum()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


X = df.drop('MEDV',axis=1)
y = df['MEDV']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)


mae = mean_absolute_error(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_pred,y_test)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

joblib.dump(model,'boston_model.pkl')
joblib.dump(scaler,'boston_scaler.pkl')
joblib.dump(X.columns.tolist(),'features_name.pkl')
