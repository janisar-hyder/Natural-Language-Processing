# Perform standard imports:
import spacy
nlp = spacy.load('en_core_web_sm')

import nltk
nltk.download('stopwords')

### Print the set of spaCy's default stop words (remember that sets are unordered):
##print(nlp.Defaults.stop_words)

print(len(nlp.Defaults.stop_words))

##"""## To see if a word is a stop word"""
##
##print(nlp.vocab['myself'].is_stop)
##
##print(nlp.vocab['mystery'].is_stop)
##
### Add the word to the set of stop words. Use lowercase!
##nlp.Defaults.stop_words.add('mystery')
##
### Set the stop_word tag on the lexeme
##nlp.vocab['mystery'].is_stop = True
##
##print(len(nlp.Defaults.stop_words))
##
##print(nlp.vocab['mystery'].is_stop)
##
##"""## To remove a stop word
##Alternatively, you may decide that `'beyond'` should not be considered a stop word.
##"""

### Remove the word from the set of stop words
##nlp.Defaults.stop_words.remove('beyond')
##
### Remove the stop_word tag from the lexeme
##nlp.vocab['beyond'].is_stop = False
##
##print(len(nlp.Defaults.stop_words))
##
##print(nlp.vocab['beyond'].is_stop)

import string
import re
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
# load data
text = 'The Quick brown fox jump over the lazy dog!'

# split into words
tokens = word_tokenize(text)
print(tokens)

# convert to lower case
tokens = [w.lower() for w in tokens]
print(tokens)

# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
print(re_punc)

# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in tokens]
print(stripped)

# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
print(words)

# filter out non-stop words
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words)

print(nlp.vocab['dog'].is_stop)


