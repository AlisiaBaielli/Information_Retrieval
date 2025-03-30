import random
import re
import unicodedata
from typing import Dict

import contractions
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet


def preprocess_text(text: str) -> str:
    """
    Preprocess the input string by lowercasing it and removing non-alphanumeric characters
    Parameters
    ----------
    text: str
        the input string
    Returns
    -------
    str
        the preprocessed string
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = contractions.fix(text)  # contractions
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def query_expansion(query: str) -> str:
    """
    Expand the query by adding synonyms for each word in the query.

    Parameters
    ----------
    query: str
        the input query string
    Returns
    -------
    str
        the expanded query string
    """
    words = query.split()
    expanded_query = list(query.split())

    for word in words:
        # find synonyms for each word
        synonyms = set()  # Initialize the set for synonyms
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        
        # Take the first synonym and add it to the expanded query
        if synonyms:
            # Remove the original word from the set and take the first synonym
            synonym = next(iter(synonyms - {word}), None)
            if synonym:
                expanded_query.append(synonym)

    return " ".join(expanded_query)


def data_noise(text: str, noise_perc: float = 0.0) -> str:
    """
    Add noise to the data by creating typos or merging the text in the query randomly.

    Parameters
    ----------
    text: str
        the input text
    noise_perc: float
        the probability of adding noise to the text
    Returns
    -------
    str
        the text with added noise

    """
    result = text
    words = text.split()
    n = len(words)

    # word merging with a probability of noise_perc
    if random.random() < noise_perc and n > 1:
        i = random.randint(0, n - 2)
        words[i] = words[i] + words[i + 1]
        words.pop(i + 1)
        result = " ".join(words)

    # typos with a probability of noise_perc
    if random.random() < noise_perc and n > 0:
        typo_index = random.randint(0, n - 2)  #  pick a word to add a typo
        typo_word = words[typo_index]

        if len(typo_word) > 3: # skip short words
            char_idx = random.randint(0, len(typo_word) - 1)
            typo_word = (
                typo_word[:char_idx]
                + chr(ord(typo_word[char_idx]) + random.randint(-1, 1))
                + typo_word[char_idx + 1 :]
            )
        words[typo_index] = typo_word
        result = " ".join(words)

    return result


# example = "Mexico, population policy life expectancy"
# example_prep = preprocess_text(example)
# print(example_prep)  # mexico population policy life expectancy
# print(query_expansion(example_prep))  # mexico population policy life expectancy
# print("*" * 50)
# print(query_expansion(example))  # hello world

# # test data_noise
# print(data_noise("hello world", 1))
# print(data_noise("good morning my love", 1))
