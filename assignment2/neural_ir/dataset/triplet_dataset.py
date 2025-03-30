from torch.utils.data import Dataset

from neural_ir.utils.dataset_utils import read_pairs, read_triplets
from neural_ir.utils.preprocess import data_noise, preprocess_text, query_expansion


class TripletDataset(Dataset):
    """
    TripletDataset for contrastive training. This dataset consists of a list of (query, positive document, negative document) triplets.
    During training, the model would be optimize to produce high scores for (query, positive) pairs and low scores for (query, negative) pairs.
    Attributes
    ----------
    collection: dict
        a dictionary that maps document id to document text
    queries: dict
        a dictionary that maps query id to query text
    triplets: list
        a list of id triplets (query_id, pos_id, neg_id)

    HINT: - make sure to implement and use the functions defined in utils/dataset_utils.py
    """

    # TODO: implement this method
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        train_triplets_path: str,
        preprocess_d: bool = False,
        preprocess_q: bool = False,
        query_noise: float = 0.0,
        query_expansion: bool = False,
    ):
        """
        Constructing a TripletDataset
        Parameters
        ----------
        collection_path: str
            path to a tsv file,  each line has a document id and document text separated by a tab character
        queries_path: str
            path to a tsv file,  each line has a query id and query text separated by a tab character
        train_triplets_path: str
            path to a tsv file,  each line has a query id, positive id and negative id separated by a tab character
        """
        # BEGIN SOLUTION
        self.query_noise = query_noise
        self.query_expansion = query_expansion
        self.collection = dict(read_pairs(collection_path))
        self.queries = dict(read_pairs(queries_path))
        self.triplets = read_triplets(train_triplets_path)

        if preprocess_d:
            print("[TripletDataset] Preprocessing documents")
            self.collection = {
                k: preprocess_text(v) for k, v in self.collection.items()
            }
        if preprocess_q:
            print("[TripletDataset] Preprocessing queries")
            self.queries = {k: preprocess_text(v) for k, v in self.queries.items()}
        # END SOLUTION

    # TODO: implement this method
    def __len__(self):
        """
        Return the number of triplets
        """
        # BEGIN SOLUTION
        return len(self.triplets)
        # END SOLUTION

    # TODO: implement this method
    def __getitem__(self, idx):
        """
        Get text contents of the idx-th triplet.
        Parameters
        ----------
        idx: int
            the index of the triplet to return
        Returns:
        tuple:
            (query_text, pos_text, neg_text)
        """
        # BEGIN SOLUTION
        query_id, pos_id, neg_id = self.triplets[idx]
        query_text = self.queries[query_id]
        pos_text = self.collection[pos_id]
        neg_text = self.collection[neg_id]
        if self.query_noise > 0:
            query_text = data_noise(query_text, self.query_noise)
        if self.query_expansion:
            query_text = query_expansion(query_text)
        
        return query_text, pos_text, neg_text
        # END SOLUTION
