import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import csv
import kfold_template

df = pd.read_csv("question1/part_1/training_data/dataset.csv")

review_text = df['reviewtext']
review_stars = df['stars']

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
  
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(review_text)

data = count_vectorize_transformer.transform(review_text)

logreg = LogisticRegression()
results = kfold_template.run_kfold(data, review_stars, logreg, 4, False, True, True)

# [0.8925233644859814, 0.9439252336448598, 0.9158878504672897, 0.8598130841121495]
# [0.897196261682243, 0.9299065420560748, 0.9392523364485982, 0.9205607476635514]

print([i[0] for i in results])
print([i[1] for i in results])

logreg.fit(data, review_stars)

# review_text = []
# # We are using csv reader to read the dataset and we are storing reviewtext and stars in two list
# with open('question1/part_1/sample_new_data/sample_new.csv') as f:
#   reader = csv.DictReader(f)
#   for row in reader:
#     review_text.append(row['reviewtext'])

# # Storing the text in a two column dataframe with one column being the review text and another being review stars
# new_reviews = pd.DataFrame(data={"text": review_text})

# new_reviews_transformed = count_vectorize_transformer.transform(new_reviews.iloc[:,0])

# prediction = logreg.predict(new_reviews_transformed)
# predictionProb = logreg.predict_proba(new_reviews_transformed)
# print(prediction)
# print(predictionProb)