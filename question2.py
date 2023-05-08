import nltk

nltk.download("stopwords")
nltk.download("wordnet")

import csv
import json
import pandas

import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
print(stopwords.words("english"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import kfold_template

import dill as pickle

profile = []
stars = []

# We are using csv reader to read the dataset and we are storing reviewtext and stars in two list
with open('question2/part_2/training_data/dataset.csv') as f:
  reader = csv.DictReader(f)
  for row in reader:
    profile.append(row['profile'])
    stars.append(int(row['stars']))

# Storing the text in a two column dataframe with one column being the review text and another being review stars
dataset = pandas.DataFrame(data={"profile": profile, "stars": stars})

print(dataset.shape)

# We don't need to do following lines since we already have separate training and test sets
'''
dataset = dataset[0:3000]
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==3)|(dataset['stars']==5)]
dataset.reset_index(drop=True, inplace=True)
print(dataset.shape)
'''


data = dataset['profile']
target = dataset['stars']

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
  text_processed = text.translate(str.maketrans('', '', string.punctuation))
  text_processed = text_processed.split()
  result = []
  for word in text_processed:
    word_processed = word.lower()
    if word_processed not in stopwords.words("english"):
      word_processed = lemmatizer.lemmatize(word_processed)
      result.append(word_processed)
  return result
  
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)

machine = MultinomialNB()

results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)

# [0.8925233644859814, 0.9439252336448598, 0.9158878504672897, 0.8598130841121495]

print([i[0] for i in results])
print([i[1] for i in results])

machine = MultinomialNB()
machine.fit(data,target)

with open("machine.pickle", "wb")  as f:
  pickle.dump(machine, f)
  pickle.dump(count_vectorize_transformer, f)
  pickle.dump(lemmatizer, f)
  pickle.dump(stopwords, f)
  pickle.dump(string, f)
  pickle.dump(pre_processing, f)