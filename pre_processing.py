from nltk.corpus import brown, wordnet, stopwords
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

"""
You may need to:
>>> nltk.download('brown')
>>> nltk.download('stopwords')
>>> nltk.download('universal_tagset')
"""

def dummy(doc):
    """Do nothing"""
    return doc

def clean(tokens):
    """ :param tokens: list of tokenized words from natural language text
        :return tokens in lowercase, all punctuations are removed
    """
    remove_string = string.punctuation + "--" +  "''" +  "``"
    clean_tokens = []
    for token in tokens:
        if token not in remove_string:
          if token not in stopwords.words('english'):
                clean_tokens.append(token.lower())
    return clean_tokens

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(universal_tag):
    """Binding the built-in pos tag of Brown to Wordnet tag.
    Reference: https://coling.epfl.ch/TP/TP-tagging.html
    """
    if universal_tag == 'VERB':
        return wordnet.VERB
    elif universal_tag == 'ADJ':
        return wordnet.ADJ
    elif universal_tag == 'ADV':
        return wordnet.ADV
    else:
        return wordnet.NOUN

# PREPARE THE CORPUS
# retrieve pos tags of 02 categories: romance, non-roman (both immaginative)
romance_tagged = brown.tagged_words(tagset="universal", categories='romance')
non_romance_tagged = brown.tagged_words(tagset="universal", categories=['adventure', 'humor', 'mystery', 'science_fiction'])

# convert universal tag to Wordnet tag; lemmatize words in the 02 document
romance_docs = []
non_roman_docs = []

for word, tag in romance_tagged:
    romance_docs.append(lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)))

for word, tag in non_romance_tagged:
    non_roman_docs.append(lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)))

# Create corpus space with clean tokens inside each document: convert to lowercase and remove all punctuations + stop-words
corpus = {}
corpus['romance'] = clean(romance_docs)
corpus['non-romance'] = clean(non_roman_docs)

# PREPARE THE TOKENS TO RANK
total_words = []
total_words.extend(corpus['romance'])
total_words.extend(corpus['non-romance'])

unique_tokens = list(set(total_words)) # make tokens unique

""" An overview of data:
>>> print('Romance before clean:', len(romance_docs))
>>> print('Non romance before clean', len(non_roman_docs))
>>> print('Romance after clean:', len(corpus['romance']))
>>> print('Non romance after clean:', len(corpus['non-romance']))
>>> print('Total tokens in the corpus:', len(total_words))
>>> print('Total unique tokens in the corpus:', len(unique_tokens))

Romance before clean: 70022
Romance after clean: 32308

Non romance before clean 162676
Non romance after clean: 77042

Total meaningful tokens in the corpus: 109350
Total unique and meaningful tokens in the corpus: 13711

"""

# Compile the tf-idf algorithm
tf_idf = TfidfVectorizer(analyzer='word', 
                        tokenizer=dummy, 
                        preprocessor=dummy,
                        token_pattern=None)


# Apply the tf-idf algorithm to create tf-idf matrix
tfs = tf_idf.fit_transform(corpus.values())

tokens_to_rank = tf_idf.transform(unique_tokens)
print(tokens_to_rank)

feature_names = tf_idf.get_feature_names()
corpus_index = [n for n in corpus]

df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)

export_csv = df.to_csv (r'tf_idf_romance_matrix.csv', header=True) 

# Reference: 
# http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/ 
# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
# https://www.nltk.org/book/ch02.html#ref-raw-access
# https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php 
# https://stackoverflow.com/questions/46597476/how-to-print-tf-idf-scores-matrix-in-sklearn-in-python
# https://sklearn.org/modules/feature_extraction.html#text-feature-extraction
