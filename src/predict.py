#!/usr/bin/python3

import sys
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import sparse

from train import extract_features

def predict(text):
    # preprocessing
    wordnet_lemmatizer = WordNetLemmatizer()
    stop = stopwords.words('english') + list(string.punctuation) + list(["``", "''", '""'])
    preprocessed = " ".join([wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop])

    # feature extraction
    clf, count_vectorizer, scaler = joblib.load("classifier.pkl")
    count_matrix = count_vectorizer.transform([preprocessed])
    engineered = extract_features([text], scaler)
    features = sparse.hstack((count_matrix, engineered.values))

    return "spam" if clf.predict(features) == [1] else "not spam" 
    
    
if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide a test sms message in quotes as an argument"
    text = sys.argv[1] #"Hello, how are you?"
    print("Got sms: %s" % text)
    print("Predicted class: %s" % predict(text))
