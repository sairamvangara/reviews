import pandas as pd
import pickle
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#load data
data = pd.read_csv("all products_reviews.csv")
data.dropna(subset=['review'], inplace=True)
data.drop('asin', axis=1, inplace=True)
data.drop('date', axis=1, inplace=True)
data.drop('rating', axis=1,inplace=True)

def analyze_sentiment(review):
    analysis = TextBlob(review)
    # Classify sentiment as Positive, Negative, or Neutral
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment'] = data['review'].apply(analyze_sentiment)

LE = preprocessing.LabelEncoder()
data.Sentiment = LE.fit_transform(data.Sentiment)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
X_train_tolist = X_train[:, 1].tolist()
vect = CountVectorizer()  # gets freq table fr tokens in all docs(comments)-Hence lemmatization applied
tfidf = TfidfTransformer()
clf = RandomForestClassifier()

# train classifier
X_train_counts = vect.fit_transform(X_train_tolist)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)  # fitting alg with vectors(by applyng countvectorization and tfidf)

X_test_tolist = X_test[:, 1].tolist()
# predict on test data
X_test_counts = vect.transform(X_test_tolist)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

pickle.dump(analyze_sentiment, open('transform.pkl','wb'))
pickle.dump(vect, open('transform.pkl', 'wb'))
pickle.dump(tfidf, open('transform1.pkl', 'wb'))

# test accuracy
print("Test accuracy")
print(clf.score(X_test_tfidf, y_test)*100)

X_test_tolist = X[:, 1].tolist()
x1 = vect.transform(X_test_tolist)
x1_tfidf = tfidf.transform(x1)
print(clf.score(x1_tfidf, y)*100)
print("x1_tfidf")
print(x1_tfidf)

pickle.dump(clf, open('model.pkl', 'wb'))
print('Success')
