import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
print(df.head(2))

#print(df.info())
#print(df.describe())

#checking missing values 
missing = df.isnull().sum()
print("Missing values are ",missing[missing > 0])


#distribution of target variable 
sns.histplot(df['age'],kde=True)
plt.title("Distribution of age in the dataset")
plt.show()


#correlation matrix 
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap="coolwarm")
plt.title("Feature correlation ")
plt.show()


#top correlated features wrt to the output features 
top_corr = corr_matrix['age'].sort_values(ascending=False)[1:6]
print("The top features correlated with age are ",top_corr)

#visualization of the best correlated features 

for feature in top_corr.index: 
	plt.figure()
	sns.scatterplot(data=df,x=feature,y="age")
	plt.title(f"{feature} vs Age")
	plt.show()