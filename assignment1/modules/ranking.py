from collections import defaultdict

import numpy as np
from .dataset import Dataset


# TODO: Implement this!
def compute_df(documents):
    """
    Compute the document frequency of all terms in the vocabulary.
    Input: A list of documents
    Output: A dictionary with {token: document frequency}
    """
    # BEGIN SOLUTION
    df = {}
    for tokens in documents:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1
    return df 
    # END SOLUTION


# TODO: Implement this!
def tfidf_tf_score(tf):
    """
    Apply the correct formula (see assignment1.ipynb file) for the term frequency of the tfidf search ranking.
    Input:
        tf - the simple term frequency, representing how many times a term appears in a document
    Output: as single value for the term frequency term after applied a formula on it
    """
    # BEGIN SOLUTION
    return np.log(1 + tf)
    # END SOLUTION


# TODO: Implement this!
def tfidf_idf_score(tdf, N):
    """
    Apply the correct formula (see assignment1.ipynb file) for the idf score of the tfidf search ranking.
    Input:
        tdf - the document frequency for a single term
        N - the total number of documents
    Output: as single value for the inverse document frequency
    """
    # BEGIN SOLUTION
    return np.log(N / tdf)
    # END SOLUTION


# TODO: Implement this!
def tfidf_term_score(tf, tdf, N):
    """
    Combine the tf score and the idf score.
    Hint: Use the tfidf_idf_score and the tfidf_tf_score functions
    Input:
        tf - the simple term frequency, representing how many times a term appears in a document
        tdf - the document frequency for a single term
        N - the total number of documents
    Output: a single value for the tfidf score
    """
    # BEGIN SOLUTION
    tf_score = tfidf_tf_score(tf)
    idf_score = tfidf_idf_score(tdf, N)
    return tf_score * idf_score
    # END SOLUTION


# TODO: Implement this!
def tfidf_search(query: str, dh: Dataset) -> list:
    """
    Perform a search over all documents with the given query using tf-idf.
    Hint: Use the tfidf_term_score method
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    df = dh.get_df()
    processed_query = dh.preprocess_query(query)
    N = dh.n_docs
    # BEGIN SOLUTION
    doc_scores = {}
    for term in processed_query:
        if term in index:
            idf_score = tfidf_idf_score(df[term], N)
            for doc_id, tf in index[term]:
                tf_score = tfidf_tf_score(tf)
                tfidf_score = tf_score * idf_score
                if doc_id in doc_scores:
                    doc_scores[doc_id] += tfidf_score
                else:
                    doc_scores[doc_id] = tfidf_score

    sorted_doc_scores = sorted(
        doc_scores.items(), key=lambda item: item[1], reverse=True
    )
    return sorted_doc_scores
    # END SOLUTION


# TODO: Implement this!
def naive_ql_document_scoring(query: str, dh: Dataset) -> list:
    """
    Perform a search over all documents with the given query using a naive QL model.
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), unsorted in relevance to the given query
    """
    index = dh.get_index()
    _doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    # BEGIN SOLUTION
    doc_scores = {}
    
    for term in processed_query:
        if term in index:
            for doc_id, term_freq in index[term]:
                term_prob = term_freq / _doc_lengths[doc_id]
                if doc_id in doc_scores:
                    doc_scores[doc_id] *= term_prob
                else:
                    doc_scores[doc_id] = term_prob
    
    return list(doc_scores.items())
    # END SOLUTION


# TODO: Implement this!
def naive_ql_document_ranking(results: list) -> list:
    """
    Sort the results.
    Input:
        result - a list of (document_id, score), unsorted in relevance to the given query
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results
    # END SOLUTION


# TODO: Implement this!
def naive_ql_search(query: str, dh: Dataset) -> list:
    """
    1. Perform a search over all documents with the given query using a naive QL model,
    using the method naive_ql_document_scoring
    2. Sort the results using the method naive_ql_document_ranking
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    unsorted_res = naive_ql_document_scoring(query, dh)
    sorted_res = naive_ql_document_ranking(unsorted_res)
    return sorted_res
    # END SOLUTION


# TODO: Implement this!
def ql_background_model(query: str, dh: Dataset) -> tuple:
    """
    Compute the background model of the smooth ql function.
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a tuple consisting of a (query, dh, collection_prob:dict with {term: collection frequency of term/collection_length})
    """
    # BEGIN SOLUTION
    index = dh.get_index()
    collection_length = sum(dh.get_doc_lengths().values())
    collection_freq = {}

    for term in index:
        term_freq = sum(freq for _, freq in index[term])
        collection_freq[term] = term_freq / collection_length

    return (query, dh, collection_freq)
    # END SOLUTION


# TODO: Implement this!
def ql_document_scoring(
    query, dh: Dataset, collection_prob: dict, smoothing: float = 0.1
) -> list: 
    """
    Perform a search over all documents with the given query using a QL model
    with Jelinek-Mercer Smoothing (with default smoothing=0.1).
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
        collection_prob - a dictionary with {term: collection frequency of term/collection_length}
        smoothing - the smoothing parameter (lambda parameter in the smooth QL equation)
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    # BEGIN SOLUTION
    doc_scores = defaultdict(float)

    for doc_id, terms in dh.doc_rep:
        for term in processed_query:
            if term in index:
                tf = terms.count(term)
                score = (
                    smoothing * collection_prob.get(term, 0)
                    + (1 - smoothing) * tf / doc_lengths[doc_id]
                )
                if score:
                    doc_scores[doc_id] += np.log(score)

    return naive_ql_document_ranking(list(doc_scores.items()))
    # END SOLUTION


# TODO: Implement this!
def ql_search(query: str, dh: Dataset, smoothing: float = 0.1) -> list:
    """
    Perform a search over all documents with the given query using a QL model
    with Jelinek-Mercer Smoothing (set smoothing=0.1).

    1. Create the background model using the method ql_background_model
    2. Perform a search over all documents with the given query using a QL model,
    using the method ql_document_scoring
    3. Sort the results using the method ql_document_ranking

    Note #1: You might have to create some variables beforehand and use them in this function

    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
        smoothing - the smoothing parameter (lambda parameter in the smooth QL equation)
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    query, dh, collection_prob = ql_background_model(query, dh)
    unsorted_res = ql_document_scoring(query, dh, collection_prob, smoothing)
    sorted_res = naive_ql_document_ranking(unsorted_res)
    return sorted_res
    # END SOLUTION


# TODO: Implement this!
def bm25_tf_score(tf, doclen, avg_doc_len, k_1, b):
    """
    Compute the bm25 tf score that uses two parts. The numerator,
    and the denominator.
    Input:
        tf - the term frequency used in bm25
        doclen - the document length of
        avg_doc_len - the average document length
        k_1 - contant of bm25
        b - constant of bm25
    Output: a single value for the tf part of bm25 score
    """
    # BEGIN SOLUTION
    numerator = tf * (k_1 + 1)
    denominator = tf + k_1 * (1 - b + b * (doclen / avg_doc_len))
    return numerator / denominator
    # END SOLUTION


# TODO: Implement this!
def bm25_idf_score(df, N):
    """
    Compute the idf part of the bm25 and return its value.
    Input:
        df - document frequency
        N - total number of documents
    Output: a single value for the idf part of bm25 score
    """
    # BEGIN SOLUTION
    return np.log((N - df + 0.5) / (df + 0.5) + 1)
    # END SOLUTION


# TODO: Implement this!
def bm25_term_score(tf, df, doclen, avg_doc_len, k_1, b, N):
    """
    Compute the term score part of the bm25 and return its value.
    Hint 1: Use the bm25_tf_score method.
    Hint 2: Use the bm25_idf_score method.
    Input:
        tf - the term frequency used in bm25
        doclen - the document length of
        avg_doc_len - the average document length
        k_1 - contant of bm25
        b - constant of bm25
        df - document frequency
        N - total number of documents
        Output: a single value for the term score of bm25
    """
    # BEGIN SOLUTION
    tf_score = bm25_tf_score(tf, doclen, avg_doc_len, k_1, b)
    idf_score = bm25_idf_score(df, N)
    return tf_score * idf_score
    # END SOLUTION


# TODO: Implement this!
def bm25_search(query, dh: Dataset):
    """
    Perform a search over all documents with the given query using BM25. Use k_1 = 1.5 and b = 0.75
    Note #1: You have to use the `get_index` (and `get_doc_lengths`) function created in the previous cells
    Note #2: You might have to create some variables beforehand and use them in this function
    Hint: You have to use the bm25_term_score method
    Input:
        query - a (unprocessed) query
        dh - instance of a Dataset
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = dh.get_index()
    df = dh.get_df()
    doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    # BEGIN SOLUTION
    k_1 = 1.5
    b = 0.75
    avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)
    N = len(doc_lengths)
    doc_scores = {}
    
    for term in processed_query:
        if term in index:
            for doc_id, term_freq in index[term]:
                term_score = bm25_term_score(term_freq, df[term], doc_lengths[doc_id], avg_doc_len, k_1, b, N)
                if doc_id in doc_scores:
                    doc_scores[doc_id] += term_score
                else:
                    doc_scores[doc_id] = term_score
    
    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
    # END SOLUTION


def tfidf_count_idf1_search(query: str, dh: Dataset) -> list:
    """
    Perform a search over all documents with the given query using tf-idf (count-idf1 version).
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    df = dh.get_df()
    processed_query = dh.preprocess_query(query)
    N = dh.n_docs
    #######################
    doc_scores = {}
    for term in processed_query:
        if term in index:
            idf_score = tfidf_idf_score(df[term], N)
            for doc_id, tf in index[term]:
                tf_score = tf  # NOTE: just the frequency of token in document d
                tfidf_score = tf_score * idf_score
                if doc_id in doc_scores:
                    doc_scores[doc_id] += tfidf_score
                else:
                    doc_scores[doc_id] = tfidf_score

    sorted_doc_scores = sorted(
        doc_scores.items(), key=lambda item: item[1], reverse=True
    )
    return sorted_doc_scores
    #######################


def tfidf_double_norm_tf_search(query: str, dh: Dataset, k: float = 0.5) -> list:
    """
    Perform a search over all documents with the given query using tf-idf (double normalization-tf).
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    df = dh.get_df()
    processed_query = dh.preprocess_query(query)
    N = dh.n_docs
    #######################
    # NOTE: max term frequency in each document
    max_tf_per_doc = {}
    for term in index:
        for doc_id, tf in index[term]:
            if doc_id not in max_tf_per_doc:
                max_tf_per_doc[doc_id] = tf
            else:
                max_tf_per_doc[doc_id] = max(max_tf_per_doc[doc_id], tf)

    doc_scores = {}
    for term in processed_query:
        if term in index:
            idf_score = tfidf_idf_score(df[term], N)
            for doc_id, tf in index[term]:
                # NOTE: double normalization tf
                tf_score = k + (1 - k) * (tf / max_tf_per_doc[doc_id])
                tfidf_score = tf_score * idf_score
                if doc_id in doc_scores:
                    doc_scores[doc_id] += tfidf_score
                else:
                    doc_scores[doc_id] = tfidf_score

    sorted_doc_scores = sorted(
        doc_scores.items(), key=lambda item: item[1], reverse=True
    )
    return sorted_doc_scores
    #######################
