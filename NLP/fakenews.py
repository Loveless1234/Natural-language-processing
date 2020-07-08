#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:20:50 2020

@author: ranjitsah
"""

#importing basic library to read files
import pandas as pd


df = pd.read_csv('/Users/ranjitsah/Documents/Data science/NLP/Fake news classification/kaggle_fake_train.csv')
df.head(5)
df.shape

df.keys()
df.drop(labels='id',axis=1,inplace=True)
df.head(2)

#Checking missings value
df.isna().sum()
df.dropna(inplace=True)

df.shape
df.reset_index(inplace=True)

#EDA
#importing  importanat librarie
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(18,9))
sns.countplot(x='label',data=df)
plt.xlabel('labels of nfake news')
plt.ylabel('counts of labels')

## Importing essential libraries for performing Natural Language Processing 
import  re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem  import WordNetLemmatizer

ps = PorterStemmer()
#wordnet = WordNetLemmatizer()
len(df['title'])

corpus = []

for i in range(0, len(df['title'])):
    review = re.sub('[^a-zA-Z]', ' ', df['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#display some text in corpus list
corpus[0:1]
    
#creating Bag of words Model
from sklearn.feature_extraction.text import  CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

#Extracting dependable variable y
y = df['label']

#spliting the data into training and testing set
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Creating Naive Bayes Multinominal model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

prediction = nb.predict(X_test)

#Accuracy and Evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(prediction,y_test))
print(classification_report(prediction,y_test))











