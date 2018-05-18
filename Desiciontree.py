import collections

import nltk
import wordcount as wordcount
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import pymongo
#ΑΝΕΒΑΣΜΑ ΑΡΧΕΙΩΝ
client = pymongo.MongoClient('localhost', 27017)
db = client['yelp']
data=db["reviews"].find()
yelp_reviews=pd.DataFrame(list(data)).sample(1000)
data2=db["restaurants"].find()
yelp_restaurant=pd.DataFrame(list(data2))
#ΚΑΝΟΥΜΕ ΤΑ DARAFRAME ΕΝΑ ΚΑΙ ΠΕΤΑΜΕ ΤΟ STARS TOY ΡΕΣΤΟΡΑΝΤ
yelp_restaurant=yelp_restaurant.drop("stars",axis=1)
yelp=pd.merge(yelp_reviews,yelp_restaurant,on="business_id",how='inner')
stemmer = SnowballStemmer("english")
yelp.shape
yelp['text_length'] = yelp['neighborhood'].apply(len)
yelp.head()
yelp.info()
yelp.describe()
''''#plt.show()
g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text_length', bins=150)
#plt.show()
sns.boxplot(x="stars", y="text_length", data=yelp)
#plt.show()
stars = yelp.groupby('stars').mean()
sns.heatmap(data=stars.corr(), annot=True)
#plt.show()'''
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
positive=0
negative=0
for n in yelp_class['stars']:
    if n==5:
        positive +=1
    elif n==1:
        negative+=1

sizes=[positive,negative]
labels =['positive','negative']
plt.pie(sizes, labels =labels, colors = ['red','blue'], shadow = True, )
plt.title("positive and negative crowd ")
#plt.show()

Counter = Counter(yelp_class['text'])


df = pd.DataFrame(columns=[yelp_class['text'],Counter.most_common()])
df.head(10)
''''Prepei na ginei merge me to dataframe kai na mpei sta feature gia na prosthesoume ena akomi '''

X = yelp_class[["text",'address','city']].astype(str).sum(axis=1)
y = yelp_class['stars']

import string
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print("arxise tifidf")
bow_transformer = TfidfVectorizer(analyzer=text_process).fit(X)
print("telow vectorizer")
len(bow_transformer.vocabulary_)
X = bow_transformer.transform(X)

print("telos tifidf vectorizer")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("split ok")

nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
print("desicion treeeeeee")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("RandomForest modelll")

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("lenear regression")
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef
y_pred = classifier.predict(X_test)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
