import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def jitter(values, sd=0.25):
    return [np.random.normal(v, sd) for v in values]

# A function for keeping only alpha-numeric characters and replacing all white space with a single space.
def clean_text(df, col):
    return df[col].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                  .apply(lambda x: re.sub('\s+', ' ', x).strip())

# Count the occurrences of "pattern" in df[col].
def count_pattern(df, col, pattern):
    df = df.copy()
    return df[col].str.count(pattern)

# Use regular expression tokenizer and keep apostrophes.
def split_on_word(text):
    # Returns a list of lists, one list for each sentence: [[word, word], [word, word, ..., word], ...].
    if type(text) is list:
        return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
    else:
        return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

# Removes stop words, numbers, short words, and lowercases text.
def normalize(tokenized_words):
    stop_words = stopwords.words('english')
    # Returns a list of lists, one list for each sentence: [[word, word], [word, word, ..., word], ...].
    return [[w.lower() for w in sent
             if (w.lower() not in stop_words)]
            for sent in tokenized_words]

def features(df):
    df = df.copy()
    df['n_questionmarks'] = count_pattern(df, 'Text', '\?')
    df['n_periods'] = count_pattern(df, 'Text', '\.')
    df['n_apostrophes'] = count_pattern(df, 'Text', '\'')
    df['first_word'] = df.text_clean.apply(lambda x: split_on_word(x)[0])
    question_words = ['what', 'how', 'why', 'is']
    for w in question_words:
        col_wc = 'n_' + w
        col_fw = 'fw_' + w
        df[col_wc] = count_pattern(df, 'text_clean', w)
        df[col_fw] = (df.first_word == w) * 1
        
    del df['first_word']
    
    df['n_words'] = df.Text.apply(lambda x: len(split_on_word(x)))
    return df

def flatten_words(list1d, get_unique=False):
    qa = [s.split() for s in list1d]
    if get_unique:
        return sorted(list(set([w for sent in qa for w in sent])))
    else:
        return [w for sent in qa for w in sent]

# Read training data
training = pd.read_csv('data/newtrain.csv')
#print(training.head())

# Read test data
test = pd.read_csv('./data/newtest.csv')
#print(test.head())

# Clean data
training['text_clean'] = clean_text(training, 'Text')
test['text_clean'] = clean_text(test, 'Text')

# TF-IDF
all_text = training['text_clean'].values.tolist() + test['text_clean'].values.tolist()
# Create vocabuary
vocab = flatten_words(all_text, get_unique=True)
# Remove stop word
tfidf = TfidfVectorizer(stop_words='english', vocabulary=vocab)
# Create training and test matrix
training_matrix = tfidf.fit_transform(training.text_clean)
test_matrix = tfidf.fit_transform(test.text_clean)

training = features(training)
training = pd.concat([training, pd.DataFrame(training_matrix.todense())], axis=1)
#print(training.head(3))
test = features(test)
test = pd.concat([test, pd.DataFrame(test_matrix.todense())], axis=1)

# Split training data
train, dev = cross_validation.train_test_split(training, test_size=0.2, random_state=1868)

# Training with SVM
svm = LinearSVC(dual=False, max_iter=5000)
features = train.columns[3:]
X = train[features].values
y = train['Category'].values
features_dev = dev[features].values

kf = cross_validation.KFold(n=len(train), n_folds=5)
print(np.array([svm.fit(X[tr], y[tr]).score(X[te], y[te]) for tr, te in kf]).mean())

# Test
svm.fit(X, y)
dev_predicted = svm.predict(features_dev)
# Calculate accuracy
accuracy = accuracy_score(dev.Category, dev_predicted)
print(accuracy)

# Show
plt.figure(figsize=(5, 4))

plt.scatter(jitter(dev.Category, 0.15),
            jitter(dev_predicted, 0.15),
            color='#348ABD', alpha=0.25)

plt.xlabel('Ground Truth')
plt.ylabel('Predicted')
plt.show()

