# Use tf-idf algorithm to rank the importance of tokens 

Disclaimer: I can hardly write the entire code just from my head. So please check the reference for their origins and appreciate those geniuses behind them.

In this specific project I'm interested in the usage of verbs in romance and non-romance fiction text. I used the term frequency-inverse document frequency (tf-idf) algorithm with sklearn in Python, and Latent Semantic Analysis (LSA) space, freely distributed in lsa.colorado.edu. 

For the math behind tf-idf, please check the reference list; for its code, please check the rank_tokens_tf_idf.py - I hope to explain the code sufficiently inside the file already. For the LSA, please check the website as I used the ready trained space there, no programming involved. A minor technique included is lemmatization, I prefer this to stemming for this project as it preserves the semantic meaning of the word.

The corpus in use is the Brown corpus of American English, incorporated in the nltk package. The text in the corpus is already tokenized and part-of-speech (pos) tagged, which made the extraction of word type, such as verb, very straightforward. 

```
> from nltk.corpus import brown
> brown.words(categories='romance')
['They', 'neither', 'liked', 'nor', 'disliked', 'the', ...]

> brown_tagged = brown.tagged_words(tagset="universal", categories='romance')
> brown_tagged[0:10]
[('They', 'PRON'), ('neither', 'CONJ'), ('liked', 'VERB'), ('nor', 'CONJ'), ('disliked', 'VERB'), ('the', 'DET'), ('Old', 'ADJ'), ('Man', 'NOUN'), ('.', '.'), ('To', 'ADP')]
```
Let's further examine the corpus for the project:

```
> from nltk.corpus import brown
> brown.categories()
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
```


As can be seen, the corpus does not specific romance text against non-romance, but it does specify informative text and imaginative text (check the reference). The imaginative prose, except for 'romance', include 'adventure', 'humor', 'mystery', and 'science_fiction', of which I combined all into the 'non-romance' categories for this project.

```
>>> len(brown.words(categories='romance'))
70022

>>> len(brown.words(categories=['adventure', 'humor', 'mystery', 'science_fiction']))
162676
```
Just to have a sense of what I am dealing with, I also converted this into readable file, include under the 'output/human_readable_corpus_convert_from_brown' directory. Let's see the what type of word we have in the two categories:
```
> romance_tagged = brown.tagged_words(tagset="universal", categories='romance')
> non_romance_tagged = brown.tagged_words(tagset="universal", categories=['adventure', 'humor', 'mystery', 'science_fiction'])
> tag_fd_romance = nltk.FreqDist(tag for (word, tag) in romance_tagged)
> tag_fd_non_romance = nltk.FreqDist(tag for (word, tag) in non_romance_tagged)

> tag_fd_romance.most_common()
[('VERB', 12784), ('NOUN', 12550), ('.', 11397), ('DET', 7211), ('ADP', 6918), ('PRON', 5748), ('ADV', 3986), ('ADJ', 3912), ('PRT', 2655), ('CONJ', 2469), ('NUM', 321), ('X', 71)]

> "{0:.0%}".format(fd_non_romance.freq('VERB')), "{0:.0%}".format(fd_non_romance.freq('NOUN')), "{0:.0%}".format(fd_non_romance.freq('ADJ'))
44% 43% 13% 

> tag_fd_non_romance.most_common()
[('NOUN', 31179), ('VERB', 28782), ('.', 25738), ('DET', 18405), ('ADP', 16667), ('PRON', 11858), ('ADV', 9372), ('ADJ', 8376), ('PRT', 6008), ('CONJ', 4998), ('NUM', 1171), ('X', 122)]
> "{0:.0%}".format(fd_non_romance.freq('VERB')), "{0:.0%}".format(fd_non_romance.freq('NOUN')), "{0:.0%}".format(fd_non_romance.freq('ADJ'))
42% 46% 12%
```

As specified in the output, there are a lot of verbs! Even though, non-romance fictional text seems to favour NOUNs a bit more.

A side not: Not really relevant but I included in this respository some piece of R code (in 'naive_r' folder) that I used to visualize and test the correlation matrix collected from the Collorado LSA. Like below picture, in case, you want to make some sence of the data. This can all be done in Python but that is for another project.
![statimage](/image/MDS_verbs.jpeg)
![statimage](/image/boxplot_verb.jpeg)

# References:
Frequency distribution tagged text: https://www.nltk.org/book/ch01.html <br>
FreqDist documentation: https://kite.com/python/docs/nltk.FreqDist <br>
Categorizing and Tagging Words: https://www.nltk.org/book/ch05.html <br>
Universal POS tags: https://universaldependencies.org/u/pos/all.html#al-u-pos/SYM and https://universaldependencies.org/u/feat/index.html <br>
TFidfVectorizer: http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/ <br>
https://www.geeksforgeeks.org/python-lemmatization-with-nltk/ <br>
https://www.nltk.org/book/ch02.html#ref-raw-access <br>
https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php <br>
https://stackoverflow.com/questions/46597476/how-to-print-tf-idf-scores-matrix-in-sklearn-in-python <br>
https://stackoverflow.com/questions/46580932/calculate-tf-idf-using-sklearn-for-n-grams-in-python <br>
https://sklearn.org/modules/feature_extraction.html#text-feature-extraction <br>
https://sklearn.org/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer <br>
