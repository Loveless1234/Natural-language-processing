#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:01:27 2020

@author: ranjitsah
"""
#importing pandas and reading dataset
import pandas as pd

df = pd.read_csv('Stock Headlines.csv',encoding = 'ISO-8859-1')
df.head(3)
df.keys()
df.shape

#importing some library for visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.countplot(x='Label',data=df)  
plt.xlabel('labels')
plt.ylabel('counts')

df.isna().any()
df.dropna(inplace=True)   

df_new = df.copy()


print(df_new.head(4))
df_new.columns

#Dividing the whole dataset into train and test part
train = df_new[df_new['Date']<'20150101']
test = df_new[df_new['Date']>'20141231']
train.columns[2:27]

y_train = train['Label']
train = train.iloc[:,2:27]

y_test = test['Label']
test = test.iloc[:,2:27]




# Removing punctuation and special character from the text
train.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)
test.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

#checking the columns
train.columns
test.columns

# Renaming columns
new_columns = [str(i) for i in range(0,25)]
train.columns = new_columns
test.columns = new_columns

# Converting the entire text to lower case
for i in new_columns:
  train[i] = train[i].str.lower()
  test[i] = test[i].str.lower()
  
  
# Joining all the columns
train_headlines = []
test_headlines = []

for row in range(0, train.shape[0]):
  train_headlines.append(' '.join(str(x) for x in train.iloc[row, 0:25]))

for row in range(0, test.shape[0]):
  test_headlines.append(' '.join(str(x) for x in test.iloc[row, 0:25]))


train_headlines[0]
test_headlines[0:5]


# Importing essential libraries for performing Natural Language Processing on given dataset
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# Creating corpus of train dataset

train_corpus = []

for i in range(0, len(train_headlines)):
  
  # Tokenizing the news-title by words
  words = train_headlines[i].split()

  # Removing the stopwords
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  headline = ' '.join(words)

  # Building a corpus of news-title
  train_corpus.append(headline)
  
  # Creating corpus of test dataset
test_corpus = []

for i in range(0, len(test_headlines)):
  
  # Tokenizing the news-title by words
  words = test_headlines[i].split()

  # Removing the stopwords
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  headline = ' '.join(words)

  # Building a corpus of news-title
  test_corpus.append(headline)
  
# Creating the Bag of Words model
from sklearn.feature_extraction.text import  CountVectorizer
cv = CountVectorizer(max_features=10000, ngram_range=(2,2))
X_train=cv.fit_transform(train_corpus).toarray()
X_test = cv.transform(test_corpus).toarray()


#model Naive Bayes
from sklearn.naive_bayes import  MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

prediction = nb.predict(X_test)

from sklearn.metrics import  classification_report,confusion_matrix,accuracy_score
print(accuracy_score(prediction,y_test))
print(classification_report(prediction,y_test))
print(confusion_matrix(prediction,y_test))


#Randomforest classifier
from sklearn.ensemble import  RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=200,criterion='entropy')
rfclf.fit(X_train,y_train)

rfclf_prediction = rfclf.predict(X_test)


print(accuracy_score(rfclf_prediction,y_test))
print(classification_report(rfclf_prediction,y_test))
print(confusion_matrix(rfclf_prediction,y_test))

#LogisticRegression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

print(accuracy_score(lr_prediction,y_test))
print(classification_report(lr_prediction,y_test))
print(confusion_matrix(lr_prediction,y_test))






