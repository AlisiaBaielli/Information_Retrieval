import os

import numpy as np
import torch

from ltr.dataset import FeatureExtraction, FeatureList


class LoggingPolicy:
    """
    A simulated logging policy for Learning to Rank (LTR) click models.

    This class loads a precomputed logging policy from a `.npz` file and provides
    methods to query document rankings and simulate user clicks based on
    relevance probabilities and position-based propensities.

    Attributes
    ----------
    dlr : np.ndarray
        Array storing document length records for queries.
    lv : np.ndarray
        Array storing learned relevance probabilities for each document.
    positions : np.ndarray
        Array storing document positions for each query.
    sorted_docids : np.ndarray
        Array storing sorted document IDs for each query.
    topk : int
        The maximum rank position considered for logging.
    propensity : torch.Tensor
        Propensity scores for ranked positions, following an inverse rank weighting.
    noise : float
        The noise factor added to relevance probabilities to simulate variability.
    """

    def __init__(
        self,
        policy_path: str = "../data/",
        noise: float = 0.05,
        topk: int = 20,
        gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.05,
        feature_extractor: FeatureExtraction = None,
    ) -> None:
        """
        Initializes the LoggingPolicy by loading a precomputed logging policy.

        Parameters
        ----------
        policy_path : str, optional
            Path to the directory containing the `logging_policy.npz` file. Default is "../data/".
        """
        # Load the precomputed policy from file
        policy = np.load(os.path.join(policy_path, "logging_policy.npz"))
        print(policy.keys())

        # Document length records per query
        self.dlr = policy["dlr"]

        # Learned relevance probabilities
        self.lv = policy["lv"]

        # Positions of documents in rankings
        self.positions = policy["positions"]

        # Sorted document IDs
        self.sorted_docids = policy["sorted_docids"]

        # Maximum number of positions to consider for logging
        self.topk = topk

        # Position-based propensity scores
        self.propensity = 1.0 / torch.arange(1, self.topk + 1, requires_grad=False)

        # Noise factor for modifying relevance probabilities
        self.noise = noise

        # Gaussian noise parameters
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std

        # FeatureExtractor for extracting BM25 and other features
        self.feature_extractor = feature_extractor

    def _query_rel_probs(self, qid: int) -> np.ndarray:
        """
        Retrieves the learned relevance probabilities for a given query.

        Parameters
        ----------
        qid : int
            Query ID.

        Returns
        -------
        np.ndarray
            An array of relevance probabilities scaled to a range of [0, 1].
        """
        # Get the start and end indices for the given query
        s_i, e_i = self.dlr[qid : qid + 2]

        # Normalize relevance scores to [0, 1]
        return self.lv[s_i:e_i] / 4.0

    def query_positions(self, qid: int) -> np.ndarray:
        """
        Retrieves the document positions for a given query.

        Parameters
        ----------
        qid : int
            Query ID.

        Returns
        -------
        np.ndarray
            Array of document positions for the given query.
        """
        # Get the start and end indices for the given query
        s_i, e_i = self.dlr[qid : qid + 2]

        # Return the corresponding document positions
        return self.positions[s_i:e_i]

    def query_sorted_docids(self, qid: int) -> np.ndarray:
        """
        Retrieves the sorted document IDs for a given query.

        Parameters
        ----------
        qid : int
            Query ID.

        Returns
        -------
        np.ndarray
            Array of sorted document IDs for the given query.
        """
        # Get the start and end indices for the given query
        s_i, e_i = self.dlr[qid : qid + 2]

        # Return the sorted document IDs
        return self.sorted_docids[s_i:e_i]

    def gather_clicks(self, qid: int) -> np.ndarray:
        """
        Simulates user clicks for a given query using position-based propensity and relevance probabilities.

        The method applies a noise factor to relevance probabilities and then uses a binomial model
        to generate simulated clicks.

        Parameters
        ----------
        qid : int
            Query ID.

        Returns
        -------
        np.ndarray
            An array representing whether each document received a click (1) or not (0).
        """
        # Get document positions for the query
        pos = self.query_positions(qid)

        rel_probs = self._query_rel_probs(qid)

        # TODO: double check with the TA
        # if self.feature_extractor is None:
        #     # Get relevance probabilities for the query
        #     rel_probs = self._query_rel_probs(qid)

        # else:
        #     print(qid, self.query_sorted_docids(qid))
        #     # NOTE: use the BM25 score as relevance probability
        #     rel_probs = np.array(
        #         [
        #             self.get_bm25_for_query_doc(qid, docid)
        #             for docid in self.query_sorted_docids(qid)
        #         ]
        #     )
        #     print(rel_probs)
        #     # normalize relevance probabilities
        #     rel_probs = (rel_probs - rel_probs.min()) / (
        #         rel_probs.max() - rel_probs.min()
        #     )
        #     print(rel_probs)

        # Identify non-relevant documents within the top-k positions
        nrel_mask = (rel_probs <= 0.5) & (pos < self.topk)

        # Identify relevant documents within the top-k positions
        rel_mask = (rel_probs > 0.5) & (pos < self.topk)

        # Decrease relevance probability for relevant documents (adding noise)
        # TODO: add gaussian noise instead of constant noise
        if self.gaussian_noise:
            rel_probs[rel_mask] -= np.random.normal(
                0, self.gaussian_noise_std, len(rel_probs[rel_mask])
            )
        else:
            rel_probs[rel_mask] -= self.noise

        # Increase relevance probability for non-relevant documents (adding noise)
        if self.gaussian_noise:
            rel_probs[nrel_mask] += np.random.normal(
                0, self.gaussian_noise_std, len(rel_probs[nrel_mask])
            )
        else:
            rel_probs[nrel_mask] += self.noise

        # Initialize propensity scores as an array of zeros
        propensity = np.zeros(rel_probs.shape[0])

        # Assign propensity values for available ranks
        propensity[: min(propensity.shape[0], self.propensity.shape[0])] = (
            self.propensity[: min(propensity.shape[0], self.propensity.shape[0])]
        )

        # Compute click probabilities as the product of propensity and relevance probability
        click_probs = propensity[pos] * rel_probs

        # Initialize clicks array with zeros
        clicks = np.zeros(2)

        # Ensure at least one document gets clicked by repeating the process until a click occurs
        while clicks.sum() == 0:
            clicks = np.random.binomial(1, np.clip(click_probs, 0, 1))

        return clicks

    def get_bm25_for_query_doc(self, qid: int, docid: int, **args) -> float:
        """
        Retrieves the BM25 score for a given query-document pair using the feature extractor.

        Parameters
        ----------
        qid : int
            Query ID.
        docid : int
            Document ID.

        Returns
        -------
        float
            The BM25 score for the given query-document pair.
        """
        # Use the feature extractor to get the features, including BM25
        features = self.feature_extractor.extract(qid, docid, **args)

        # Return the BM25 score (FeatureList.f1 corresponds to BM25)
        return features[FeatureList.f1]
