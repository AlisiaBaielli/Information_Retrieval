import os
import numpy as np
from modules.utils import write_trec_run


def read_qrels(root_folder="./datasets/"):
    """
    Reads the qrels.text file.
    Output: A dictionary: query_id -> [list of relevant documents]
    """
    with open(os.path.join(root_folder, "qrels.text")) as reader:
        lines = reader.readlines()

    from collections import defaultdict

    relevant_docs = defaultdict(set)
    for line in lines:
        query_id, doc_id, _, _ = line.split()
        relevant_docs[str(int(query_id))].add(doc_id)
    return relevant_docs


# TODO: Implement this!
def precision_k(results, relevant_docs, k):
    """
    Compute Precision@K
    Input:
        results: A sorted list of 2-tuples (document_id, score),
                with the most relevant document in the first position
        relevant_docs: A set of relevant documents.
        k: the cut-off
    Output: Precision@K
    """
    if k > len(results):
        k = len(results)
    # BEGIN SOLUTION
    if k == 0:
        return 0
    results_k = [doc_id for doc_id, _ in results[:k]]
    num_relevant = len(set(results_k).intersection(relevant_docs))
    return num_relevant / k
    # END SOLUTION


# TODO: Implement this!
def recall_k(results, relevant_docs, k):
    """
    Compute Recall@K
    Input:
        results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
        relevant_docs: A set of relevant documents.
        k: the cut-off
    Output: Recall@K
    """
    # BEGIN SOLUTION
    if k == 0:
        return 0
    results_k = [doc_id for doc_id, _ in results[:k]]
    num_relevant = len(set(results_k).intersection(relevant_docs))
    return num_relevant / len(relevant_docs)
    # END SOLUTION


# TODO: Implement this!
def average_precision(results, relevant_docs):
    """
    Compute Average Precision (for a single query - the results are
    averaged across queries to get MAP in the next few cells)
    Hint: You can use the recall_k and precision_k functions here!
    Input:
        results: A sorted list of 2-tuples (document_id, score), with the most
                relevant document in the first position
        relevant_docs: A set of relevant documents.
    Output: Average Precision
    """
    # BEGIN SOLUTION
    relevant_precisions = []

    for k, (doc_id, _) in enumerate(results, start=1):
        if doc_id in relevant_docs:
            relevant_precisions.append(precision_k(results, relevant_docs, k))

    return np.mean(relevant_precisions) if relevant_precisions else 0.0
    # END SOLUTION

# TODO: Implement this!
def relevance_probability(g, g_max=1):
    """
    Compute the relevance probability (i.e. function) from the relevance grade.
    Input:
    g: The relevance grade (integer)
    g_max: The maximum possible relevance grade (integer) (default=1)
    Output: The relevance probability (float)
    """
    # BEGIN SOLUTION
    return (2 ** g - 1) / (2 ** g_max)
    # END SOLUTION


# TODO: Implement this!
def err(results, relevant_docs):
    """
    Compute the expected reciprocal rank.
    Hint 1:
    https://www.researchgate.net/publication/220269787_Expected_reciprocal_rank_for_gra
    ded_relevance
    Hint 2: Use your already implemented relevance_probability function
    Input:
    results: A sorted list of 2-tuples (document_id, score), with the most
    relevant document in the first position
    relevant_docs: A set of relevant documents.
    Output: ERR
    """
    # BEGIN SOLUTION
    err_score = 0.0
    stop_probability = 1.0

    for rank, (doc_id, _) in enumerate(results, start=1):
        g = 1 if doc_id in relevant_docs else 0
        R = relevance_probability(g)
        err_score += stop_probability * R / rank
        stop_probability *= (1 - R)

    return err_score

    # END SOLUTION

def evaluate_search_fn(
    method_name, search_fn, metric_fns, dh, queries, qrels, index_set=None
):
    # build a dict query_id -> query
    queries_by_id = dict((q[0], q[1]) for q in queries)

    metrics = {}
    for metric, metric_fn in metric_fns:
        metrics[metric] = np.zeros(len(qrels), dtype=np.float32)

    q_results = {}
    for i, (query_id, relevant_docs) in enumerate(qrels.items()):
        query = queries_by_id[query_id]
        results = search_fn(query, dh)

        q_results[query_id] = results
        for metric, metric_fn in metric_fns:
            metrics[metric][i] = metric_fn(results, relevant_docs)

    write_trec_run(q_results, f"{method_name}_{index_set}.trec")

    final_dict = {}
    for metric, metric_vals in metrics.items():
        final_dict[metric] = metric_vals.mean()

    return final_dict
