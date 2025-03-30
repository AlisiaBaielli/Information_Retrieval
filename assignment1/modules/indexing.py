# TODO: Implement this!
def build_tf_index(documents):
    """
    Build an inverted index that maps tokens to inverted lists. The output is a dictionary which takes a token
    and returns a list of (document_id, count) tuples, where 'count' is the count of the 'token' in 'document_id'
    Input: a list of documents: (document_id, tokens)
    Output: An inverted index: [token] -> [(document_id, token_count)]
    """
    # BEGIN SOLUTION
    index = {}
    for doc_id, tokens in documents:
        token_counts = {}
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        for token, count in token_counts.items():
            if token in index:
                index[token].append((doc_id, count))
            else:
                index[token] = [(doc_id, count)]
    return index

    # END SOLUTION
