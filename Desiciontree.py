import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem.snowball import SnowballStemmer
import numpy
import  string
import pymongo
#ΑΝΕΒΑΣΜΑ ΑΡΧΕΙΩΝ
client = pymongo.MongoClient('localhost', 27017)
db = client['yelp']
data=db["reviews"].find()
yelp_reviews=pd.DataFrame(list(data)).sample(10000)
data2=db["restaurants"].find()
yelp_restaurant=pd.DataFrame(list(data2))
#ΚΑΝΟΥΜΕ ΤΑ DARAFRAME ΕΝΑ ΚΑΙ ΠΕΤΑΜΕ ΤΟ STARS TOY ΡΕΣΤΟΡΑΝΤ
yelp_restaurant=yelp_restaurant.drop("stars",axis=1)
yelp=pd.merge(yelp_reviews,yelp_restaurant,on="business_id",how='inner')
stemmer = SnowballStemmer("english")
yelp.shape

print(yelp.head(5))

yelp['text_length'] = yelp['text'].apply(len)
yelp.head()
yelp.info()
yelp.describe()
g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text_length', bins=150)
plt.show()
sns.boxplot(x="stars", y="text_length", data=yelp)
plt.show()
stars = yelp.groupby('stars').mean()
sns.heatmap(data=stars.corr(), annot=True)
plt.show()
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
print(yelp_class)
print(yelp_class.shape)


X = yelp_class['text'].apply(lambda x: [stemmer.stem(y) for y in x])
y = yelp_class['stars']

import string


def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print("kanei tikenize")
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
print("telow vectorizer")
len(bow_transformer.vocabulary_)
X = bow_transformer.transform(X)
print (X.todense())
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(X)
print ("IDF:", tfidf.idf_)
X = tfidf.transform(X)
print (X.todense())
print("vocabulary",)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
