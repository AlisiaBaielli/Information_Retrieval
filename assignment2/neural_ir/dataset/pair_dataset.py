import json

from ir_measures import read_trec_run
from torch.utils.data import Dataset

from neural_ir.utils.dataset_utils import read_pairs
from neural_ir.utils.preprocess import preprocess_text, data_noise, query_expansion


class PairDataset(Dataset):
    """
    PairDataset stores pairs of query and document needed to be score in the re-ranking step.
    Attributes
    ----------
    collection: dict
        a dictionary maps document id to text
    queries: dict
        a dictionary maps query id to text
    pairs: list
        a list of (query, document) pairs for re-ranking
    qrels: dict
        a dictionary storing the ground-truth query-document relevancy. Schema:
        {
            query_id: {
                doc_id: relevance,
            ...},
            ...
        }
    top_k: int
        number of documents to rerank per query; only the first `top_k` documents will be considered for each query

    HINT: - make sure to implement and use the functions defined in utils/dataset_utils.py
          - check out the documentation of ir_measures at https://ir-measur.es/en/latest/
          - the read_trec_run method returns a generator that yields the following object:
            `yield ScoredDoc(query_id=query_id, doc_id=doc_id, score=float(score))`
            (i.e., you can use pair.query_id and pair.doc_id by iterating through the generator)

    """

    # TODO: implement this method
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        query_doc_pair_path: str,
        qrels_path: str = None,
        top_k: int = 100,
        preprocess_d: bool = False,
        preprocess_q: bool = False,
        query_noise: float = 0.0,
        query_expansion: bool = False,
    ):
        """
        Constructing PairDataset
        Parameters
        ----------
        collection_path: str
            path to a tsv file where each line store document id and text separated by a tab character
        queries_path: str
            path to a tsv file where each line store query id and text separated by a tab character
        query_doc_pair_path: str
            path to a trec run file (containing query-doc pairs) to re-rank
        qrels_path: str (optional)
            path to a qrel json file expected be formated as per the schema mentioned in the class docstring
        top_k: int (optional)
            number of documents to rerank per query; only the first `top_k` documents will be considered for each query
        """
        # BEGIN SOLUTION
        self.query_noise = query_noise
        self.query_expansion = query_expansion
        self.collection = dict(read_pairs(collection_path))
        self.queries = dict(read_pairs(queries_path))

        if preprocess_d:
            print("[PairDataset] Preprocessing documents")
            self.collection = {
                k: preprocess_text(v) for k, v in self.collection.items()
            }
        if preprocess_q:
            print("[PairDataset] Preprocessing queries")
            self.queries = {k: preprocess_text(v) for k, v in self.queries.items()}

        temp_pairs = list(read_trec_run(query_doc_pair_path))
        self.pairs = [(temp_p.query_id, temp_p.doc_id) for temp_p in temp_pairs]
        self.qrels = None
        if qrels_path:
            with open(qrels_path) as f:
                self.qrels = json.load(f) or None
        self.top_k = top_k
        # END SOLUTION

    # TODO: implement this method
    def __len__(self):
        """
        Return the number of pairs to re-rank
        """
        # BEGIN SOLUTION
        return len(self.pairs)
        # END SOLUTION

    # TODO: implement this method
    def __getitem__(self, idx):
        """
        Return the idx-th pair of the dataset in the format of (qid, docid, query_text, doc_text)
        """
        # BEGIN SOLUTION
        query_id, doc_id = self.pairs[idx]
        query_text = self.queries[query_id]
        if self.query_noise > 0:
            query_text = data_noise(query_text, self.query_noise)
        if self.query_expansion:
            query_text = query_expansion(query_text)

        return (
            query_id,
            doc_id,
            query_text,
            self.collection[doc_id],
        )
        # END SOLUTION
