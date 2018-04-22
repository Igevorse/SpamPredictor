#!/usr/bin/python3

import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from scipy import sparse


def extract_features(texts, scaler=None):
    def get_features(text):
        stop = stopwords.words('english') + list(string.punctuation) + list(["``", "''", '""'])
        wordnet_lemmatizer = WordNetLemmatizer()
        snowball_stemmer = SnowballStemmer("english")
        tokens = [snowball_stemmer.stem(w) for w in word_tokenize(text) if w not in stop]
        wordnet_tokens = [wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop]
        f = []
        f.append(len(tokens)) # number of words
        f.append(sum([1 for word in wordnet_tokens if word.isupper()])) # number of upper words
        f.append(sum([1 for word in wordnet_tokens if word.islower()])) # number of lower words
        f.append(text.count("0")) # number of 0
        f.append(text.count("1")) # number of 0
        f.append(text.count("£")) # number of £
        f.append(len([1 for word in tokens if word.isnumeric()])) # number of 0-9 WORDS
        f.append(min([len(word) for word in tokens])) # min word length
        f.append(max([len(word) for word in tokens])) # max word length
        f.append(sum([len(word) for word in tokens])) # number of symbols
        f.append(np.mean([len(word) for word in tokens])) # mean number of symbols
        f.append(sum([1  for word in tokens for letter in word if letter.isalpha()])) 

        f.append(sum([1  for word in tokens for letter in word if letter.isnumeric() and letter != '0' and letter != '1' ])) # number of digits
        f.append(sum([1  for word in wordnet_tokens for letter in word if letter.isupper()])) # number of upper letters

        f.append(sum([1  for word in wordnet_tokens for letter in word if letter.islower()])) # number of lower letters
        usual = ["sex", "won", "winner", "xchat", "free", "xxx", "18+", "cash", "prize", "urgent", "call", "you", "congratulations", "congrats"]
        f.append(sum([1 for word in tokens if word.lower() in usual])) # keywords
        return f

    additional_features = []
    for row in texts:
        additional_features.append(get_features(row))

    engineered_columns = np.array([
            '# of words',
            "# of upper words",
            "# of lower words",
             "# of 0",
             "# of 1",
             "# of £", 
             "# of 0-9 WORDS",  
             "min word length", 
             "max word length", 
             '# of symbols', 
             'mean number of symbols', 
             '# of a-z', 
             '# of 0-9', 
             '# of upper', 
             "# of lower", 
             '# of spam words'])

    engineered = pd.DataFrame(additional_features, columns=engineered_columns)

    # Extract best features and normalize. See research.ipynb for more details
    feature_mask = [
            True, #'# of words',
            True, #"# of upper words",
            True, #"# of lower words",
            True, #"# of 0",
            True, #"# of 1",
            False, # "# of £",
            True, # "# of 0-9 WORDS",  
            False, #"min word length", 
            True, # "max word length", 
            False, # '# of symbols', 
            True, # 'mean number of symbols',
            True, # '# of a-z', 
            True, # '# of 0-9', 
            False, # '# of upper', 
            False, # "# of lower", 
            True, # '# of spam words'
    ]
    features = engineered[engineered.columns[feature_mask]]
    if scaler is not None:
        features = scaler.transform(features)
    return pd.DataFrame(features, columns=engineered.columns[feature_mask])
    

def train():
    data = pd.read_csv('../data/sms.csv', encoding = "ISO-8859-1", index_col=0)

    # Drop duplicates
    data = data.drop_duplicates().reset_index(drop=True)
    # Convert labels to [0, 1]
    data["label"] = data["label"].map(lambda x: 1 if x == "spam" else 0)

    # Tokenize data
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = []
    sentences = []
    stop = stopwords.words('english') + list(string.punctuation) + list(["``", "''", '""'])

    for row in data.iterrows():
        text = row[1][0]
        tok = [wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop]
        tokens.append(tok)
        sentences.append(" ".join(tok))

    # Get feature matrices
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(sentences)


    # Scale data
    engineered_minmax = extract_features(data["text"].values)
    
    scaler = MinMaxScaler()
    scaler.fit(engineered_minmax)
    engineered_minmax = scaler.transform(engineered_minmax)
    dataset = sparse.hstack((X_counts, engineered_minmax))

    # Get the best parameters of the model
    clf_parameters = {"alpha":np.linspace(0.001, 100, 1000)}
    clf = MultinomialNB()
    gs = GridSearchCV(clf, clf_parameters, scoring=make_scorer(f1_score, pos_label=0), cv=5, n_jobs=-1)
    baseline_clf = gs.fit(dataset, data["label"])


    # Now train the model on the whole data
    clf = MultinomialNB(**baseline_clf.best_params_)
    clf.fit(dataset, data["label"])

    joblib.dump([clf, count_vect, scaler], "classifier.pkl")
    
if __name__ == "__main__":
    print("Training...", end='')
    train()
    print("finished!")
