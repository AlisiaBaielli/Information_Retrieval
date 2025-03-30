import os
from .preprocessing import process_text


def read_cacm_docs(root_folder="./datasets/"):
    """
    Reads in the CACM documents. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (doc_id, document), where 'document' is a single string created by
        appending the title and abstract (separated by a "\n").
        In case the record doesn't have an abstract, the document is composed only by the title
    """
    with open(os.path.join(root_folder, "cacm.all")) as reader:
        lines = reader.readlines()

    doc_id, title, abstract = None, None, None

    docs = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith(".I"):
            if doc_id is not None:
                docs.append((doc_id, title, abstract))
                doc_id, title, abstract = None, None, None

            doc_id = line.split()[-1]
            line_idx += 1
        elif line.startswith(".T"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            title = "\n".join(temp_lines).strip("\n")
        elif line.startswith(".W"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            abstract = "\n".join(temp_lines).strip("\n")
        else:
            line_idx += 1

    docs.append((doc_id, title, abstract))

    p_docs = []
    for did, t, a in docs:
        if a is None:
            a = ""
        p_docs.append((did, t + "\n" + a))
    return p_docs


def read_queries(root_folder="./datasets/"):
    """
    Reads in the CACM queries. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (query_id, query)
    """
    with open(os.path.join(root_folder, "query.text")) as reader:
        lines = reader.readlines()

    query_id, query = None, None

    queries = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith(".I"):
            if query_id is not None:
                queries.append((query_id, query))
                query_id, query = None, None

            query_id = line.split()[-1]
            line_idx += 1
        elif line.startswith(".W"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            query = "\n".join(temp_lines).strip("\n")
        else:
            line_idx += 1

    queries.append((query_id, query))
    return queries


def doc_lengths(documents):
    _doc_lengths = {doc_id: len(doc) for (doc_id, doc) in documents}
    return _doc_lengths


class Dataset:

    def __init__(self, n_docs, docs_by_id, doc_rep, df, tfi, stop_words, config):
        self.n_docs = n_docs
        self.docs_by_id = docs_by_id
        self.doc_rep = doc_rep
        # Document length
        self.dls = doc_lengths(doc_rep)
        # Document frequencies
        self.df = df
        self.tf_idxs = tfi
        self.stop_words = stop_words
        self.config = config

    def get_df(self):
        return self.df

    def get_doc_lengths(self):
        return self.dls

    def get_index(self):
        return self.tf_idxs

    def preprocess_query(self, query):
        """
        This function preprocesses the text given the index set, according to the specified config
        :param query:
        :param index_set:
        :return:
        """
        return process_text(query, self.stop_words, **self.config)
