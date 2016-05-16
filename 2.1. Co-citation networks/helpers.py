import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_selection import chi2
import numpy as np

wordnet = nltk.WordNetLemmatizer()
stoplist = stopwords.words('english')


def normalize_token(token):
    """
    Convert token to lowercase, and stem using the Porter algorithm.

    Parameters
    ----------
    token : str

    Returns
    -------
    token : str
    """
    return wordnet.lemmatize(token.lower())


def filter_token(token):
    """
    Evaluate whether or not to retain ``token``.

    Parameters
    ----------
    token : str

    Returns
    -------
    keep : bool
    """
    token = token.lower()
    return token not in stoplist and token.isalpha() and len(token) > 2


def extract_keywords(wordcounts_per_document, selector, n=20):
    document_index = {}    # Maps int -> fileid (str).
    vocabulary = {}    # Maps int -> word (str).
    lookup = {}    # Reverse map for vocabulary (word (str) -> int).

    I = []
    J = []
    data = []
    labels = []

    for i, (key, counts) in enumerate(wordcounts_per_document.iteritems()):
        document_index[i] = key
        for token, count in counts.iteritems():
            if count < 3:
                continue
            j = lookup.get(token, len(vocabulary))
            vocabulary[j] = token
            lookup[token] = j

            I.append(i)
            J.append(j)
            data.append(count)
        labels.append('focal' if selector(key) else 'other')

    sparse_matrix = sparse.coo_matrix((data, (I, J)))

    keyness, _ = chi2(sparse_matrix, labels)
    ranking = np.argsort(keyness)[::-1]
    _, words = zip(*sorted(vocabulary.items(), key=lambda i: i[0]))
    words = np.array(words)
    keywords = words[ranking]
    return keywords[:n]
