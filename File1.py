import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("D:\\University\\Semester 5\\NLP\\New folder\\Pos.txt", "r", encoding="utf-8").read()
short_neg = open("D:\\University\\Semester 5\\NLP\\New folder\\Neg.txt", "r", encoding="utf-8").read()
short_neutral = open("D:\\University\\Semester 5\\NLP\\New folder\\Neutral.txt", "r", encoding="utf-8").read()
 
# move this up here
all_words = []
documents = []
import spacy
nlp = spacy.load("en_core_web_sm")

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = nlp(p)
    tokens = [token for token in words if not (token.is_stop or not token.is_alpha) ]
    lemmas = [token.lemma_ for token in tokens]
    LLC = [w.lower() for w in lemmas]
    for w in LLC:
        all_words.append(w)

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = nlp(p)
    tokens = [token for token in words if not (token.is_stop or not token.is_alpha)]
    lemmas = [token.lemma_ for token in tokens]
    LLC = [w.lower() for w in lemmas]
    for w in LLC:
        all_words.append(w)

for p in short_neutral.split('\n'):
    documents.append((p, "neu"))
    words = nlp(p)
    tokens = [token for token in words if not (token.is_stop or not token.is_alpha)]
    lemmas = [token.lemma_ for token in tokens]
    LLC = [w.lower() for w in lemmas]
    for w in LLC:
        all_words.append(w)

save_documents = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:2000]
print(word_features)

save_word_features = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

featuresets_save = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\featuresets.pickle", "wb")
pickle.dump(featuresets, featuresets_save)
featuresets_save.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

# Train and evaluate Original Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

true_labels = [label for (features, label) in testing_set]
predicted_labels = [classifier.classify(features) for (features, label) in testing_set]

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("Naive Bayes - Precision:", precision)
print("Naive Bayes - Recall:", recall)
print("Naive Bayes - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Train and evaluate MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

predicted_labels = [MNB_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("MultinomialNB - Precision:", precision)
print("MultinomialNB - Recall:", recall)
print("MultinomialNB - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# Train and evaluate BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

predicted_labels = [BernoulliNB_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("BernoulliNB - Precision:", precision)
print("BernoulliNB - Recall:", recall)
print("BernoulliNB - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

# Train and evaluate Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=200))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

predicted_labels = [LogisticRegression_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("LogisticRegression - Precision:", precision)
print("LogisticRegression - Recall:", recall)
print("LogisticRegression - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# Train and evaluate LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=2000))
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

predicted_labels = [LinearSVC_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("LinearSVC - Precision:", precision)
print("LinearSVC - Recall:", recall)
print("LinearSVC - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# Train and evaluate NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

predicted_labels = [NuSVC_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("NuSVC - Precision:", precision)
print("NuSVC - Recall:", recall)
print("NuSVC - F1 Score:", f1)

# Train and evaluate SGDClassifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

predicted_labels = [SGDC_classifier.classify(features) for (features, label) in testing_set]
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print("SGDClassifier - Precision:", precision)
print("SGDClassifier - Recall:", recall)
print("SGDClassifier - F1 Score:", f1)

save_classifier = open("C:\\Users\\Raora\\Downloads\\NLP Task 4\\SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()
