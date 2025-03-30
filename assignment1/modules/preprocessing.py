import os

from nltk.tokenize import WordPunctTokenizer
from nltk import PorterStemmer


def load_stopwords(root_folder="./datasets/"):
    """
    Loads the stopwords. The dataset is assumed to be in the folder "./datasets/" by default
    Output: A set of stopwords
    """
    with open(os.path.join(root_folder, "common_words")) as reader:
        lines = reader.readlines()
    stopwords = set([l.strip().lower() for l in lines])
    return stopwords


# TODO: Implement this!
def tokenize(text):
    """
    Tokenizes the input text using nltk's WordPunctTokenizer
    Input: text - a string
    Output: a list of tokens
    """
    # BEGIN SOLUTION
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens
    # END SOLUTION


# TODO: Implement this!
def stem_token(token):
    """
    Stems the given token using the PorterStemmer from the nltk library
    Input: a single token
    Output: the stem of the token
    """
    # BEGIN SOLUTION
    stemmer = PorterStemmer()
    return stemmer.stem(token)
    # END SOLUTION


def process_text(
    text, stop_words, stem=False, remove_stopwords=False, lowercase_text=False
):
    """
    The following function puts it all together. Given a string, it tokenizes
    it and processes it according to the flags that you set.
    :param text:
    :param stop_words:
    :param stem:
    :param remove_stopwords:
    :param lowercase_text:
    :return:
    """
    tokens = []
    for token in tokenize(text):
        if remove_stopwords and token.lower() in stop_words:
            continue
        if stem:
            token = stem_token(token)
        if lowercase_text:
            token = token.lower()
        tokens.append(token)

    return tokens
