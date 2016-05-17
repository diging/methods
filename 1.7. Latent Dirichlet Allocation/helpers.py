import nltk
from nltk.corpus import stopwords

wordnet = nltk.WordNetLemmatizer()
stemmer = nltk.SnowballStemmer('english')
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
