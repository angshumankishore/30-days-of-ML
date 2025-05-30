import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder
# 2. Load data
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

sns.pairplot(df[['Age','Annual Income (k$)','Spending Score (1-100)']])
plt.show()
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Feature Correlation ")
plt.show()


X = df[['Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(1,11): 

	model = KMeans(n_clusters=k,random_state=42)
	model.fit(X_scaled)
	inertia.append(model.inertia_)


plt.plot(range(1,11),inertia,marker='o')	
plt.title("Elbow Method ")
plt.xlabel('Number of Clusters ')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=6,random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Clusters'] = clusters
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Clusters', palette='Set1')
plt.title("Customer Segments")
plt.show()

joblib.dump(kmeans,'kmeans_model.pkl')
joblib.dump(scaler,'kmeans_scaler.pkl')
