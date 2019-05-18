from nltk.corpus import brown, wordnet, stopwords, names
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

def contain_unwanted_char(word):
    """:param word: certain token as string type
       :return TRUE or FALSE
    """
    for char in word:
        if char in string.punctuation:
            return True
        elif char in string.digits:
            return True
    return False

def clean(tokens):
    """ :param tokens: list of tokenized words
        :return remove Stop-words, convert tokens in lowercase
    """
    clean_tokens = []
    for token in tokens:
        if token in stopwords.words('english'):
            continue
        elif contain_unwanted_char(token):
            continue
        elif token in names.words():
            continue
        else:
            clean_tokens.append(token)
    return clean_tokens

lemmatizer = WordNetLemmatizer()

# PREPARE THE CORPUS
# retrieve words tagged as verb, noun and adjective of 02 categories: romance, non-roman (both immaginative)
romance_tagged = [word for word, tag in brown.tagged_words(tagset="universal", categories='romance') 
                  if tag == 'VERB']
non_romance_tagged = [word for word, tag in brown.tagged_words(tagset="universal", categories=['adventure', 'humor', 'mystery', 'science_fiction']) 
                  if tag == 'VERB']

# convert Universal to Wordnet tag; then lemmatize words in the 02 document
romance_docs = []
non_romance_docs = []

for word in romance_tagged:
    romance_docs.append(lemmatizer.lemmatize(word.lower(), pos='v')) # if not lowercase lematization wont work

for word in non_romance_tagged:
    non_romance_docs.append(lemmatizer.lemmatize(word.lower(), pos='v'))

"""
As can be seen, only one verbs are extracted, tagged as 'VERB' in Brown, so when lemmatizer is employed, I can
specify the pos is 'v', meaning verb with confidence. This step can be generalized, in case multiple type of tags
are extracted,

romance_tagged = [(word, tag) for word, tag in brown.tagged_words(tagset="universal", categories='romance') 
                  if tag == 'VERB' OR tag == 'NOUN']

....by adding this function:

def get_wordnet_pos(universal_tag):
    '''Binding the built-in pos tag of Brown to Wordnet tag.
    Reference: https://coling.epfl.ch/TP/TP-tagging.html
    '''
    if universal_tag == 'VERB':
        return wordnet.VERB
    elif universal_tag == 'ADJ':
        return wordnet.ADJ
    elif universal_tag == 'ADV':
        return wordnet.ADV
    else:
        return wordnet.NOUN

and then specify accordingly in lemmatize(), for example:
for word, tag in non_romance_tagged:
    non_romance_docs.append(lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(universal_tag)))

The get_wordnet_pos() with translate the correct pos, which is important for lemmatizer to work, so it can transform
abandonning to abandon, for example; otherwise, it treats everything as noun and abandonning stays the same, although
it can actually be the case - that abandonning is actually the noun in the context. 
"""

# Create corpus space with clean tokens inside each document: convert to lowercase and remove all punctuations + stop-words
corpus = {}
corpus['romance'] = clean(romance_docs)
corpus['non_romance'] = clean(non_romance_docs)

# PREPARE THE TOKENS TO RANK
total_words = []
total_words.extend(corpus['romance'])
total_words.extend(corpus['non_romance'])

unique_tokens = sorted(list(set(total_words))) # make tokens unique

""" 
See how much we clean our data:
>>> print('Romance raw:', len(romance_docs))
>>> print('Romance processed:', len(corpus['romance']))
>>> print('Non romance raw', len(non_romance_docs))
>>> print('Non romance processed:', len(corpus['non_romance']))
>>> print('Total tokens in the corpus:', len(total_words))
>>> print('Total unique tokens in the corpus:', len(unique_tokens))

Romance raw: 12784
Romance processed: 8801

Non romance raw 28782
Non romance processed: 20601

Total tokens in the corpus: 29402
Total unique tokens in the corpus: 2608
"""

# Compile the tf-idf algorithm
tf_idf = TfidfVectorizer(vocabulary=unique_tokens, 
                        analyzer='word', 
                        tokenizer=dummy, 
                        preprocessor=dummy,
                        token_pattern=None)


# Apply the tf-idf algorithm to create tf-idf matrix
tfs = tf_idf.fit_transform(corpus.values())

# Using Pandas to export the tf-idf matrix to a .csv file for inspection
feature_names = tf_idf.get_feature_names()
corpus_index = [n for n in corpus]

df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)

export_csv = df.to_csv (r'output/tf_idf_romance_matrix_v4.csv', header=True) 

