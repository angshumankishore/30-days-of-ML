from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 
import joblib

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 0
true['label'] = 1 

df = pd.concat([fake,true],ignore_index=True)
#only keeping relevant data 

df = df[['text','label']]
print(df.head())


#spliting the data 
X = df['text']
y = df['label']

 #train test split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Tf-IDF vectorization 

vectorizer = TfidfVectorizer(stop_words='english',max_df = 0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#importing and using model 
model = MultinomialNB()
model.fit(X_train_vec,y_train)
y_pred = model.predict(X_test_vec)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report: \n",classification_report(y_test,y_pred))

joblib.dump(model,'fake_news_model.pkl')
joblib.dump(vectorizer,'tfidf_vectorizer.pkl')



