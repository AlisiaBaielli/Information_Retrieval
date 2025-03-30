import itertools

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def pointwise_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error (MSE) regression loss.

    Parameters
    ----------
    output : torch.Tensor
        Predicted values of shape [N, 1].
    target : torch.Tensor
        Ground truth values of shape [N].

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    assert target.dim() == 1
    assert output.size(0) == target.size(0)
    assert output.size(1) == 1

    ## BEGIN SOLUTION
    return F.mse_loss(output.view(-1), target.view(-1))
    ## END SOLUTION


def pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise loss for a single query.

    The loss is calculated for all possible orderings in a query using sigma=1.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1], where N is the number of <query, document> pairs.
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor or None
        Mean pairwise loss if N >= 2, otherwise None.
    """
    if labels.size(0) < 2:
        return None

    ## BEGIN SOLUTION
    sigma = 1.0
    scores = scores.view(-1)  
    labels = labels.view(-1)
    pairwise_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
    S_ij = (labels.unsqueeze(1) - labels.unsqueeze(0)).sign()

    P_ij = torch.sigmoid(sigma * pairwise_diffs)
    P_bar = 0.5 * (1 + S_ij)

    loss_m = -P_bar * torch.log(P_ij + 1e-12) - (1 - P_bar) * torch.log(1 - P_ij + 1e-12)
    mask = torch.ones_like(loss_m, device=labels.device) - torch.eye(loss_m.shape[0], device=labels.device)
    loss = (loss_m * mask).sum() / mask.sum()

    return loss
    ## END SOLUTION



def compute_lambda_i(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes lambda_i using the LambdaRank approach (sigma=1).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1], where N is the number of <query, document> pairs.
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        Lambda updates of shape [N, 1].
    """
    ## BEGIN SOLUTION
    sigma = 1.0
    scores = scores.view(-1)
    labels = labels.view(-1)

    i, j = torch.triu_indices(labels.size(0), labels.size(0), offset=1)
    S_ij = torch.sign(labels[i] - labels[j])
    s_i = scores[i]
    s_j = scores[j]
    lambda_ij = sigma * (0.5 * (1 - S_ij) - torch.sigmoid(-sigma * (s_i - s_j)))
    lambda_i = torch.zeros_like(scores)
    lambda_i.index_add_(0, i, lambda_ij)
    lambda_i.index_add_(0, j, -lambda_ij)

    return lambda_i.view(-1, 1)
    ## END SOLUTION



def mean_lambda(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean and squared mean of LambdaRank updates.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        Tensor containing mean and squared mean lambda values.
    """
    return torch.stack(
        [
            compute_lambda_i(scores, labels).mean(),
            torch.square(compute_lambda_i(scores, labels)).mean(),
        ]
    )


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the LambdaRank loss (sigma=1).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        LambdaRank loss of shape [N, 1].
    """

    ## BEGIN SOLUTION

    scores = scores.view(-1)

    lambdai = torch.zeros_like(scores)

    if labels.size(0) < 2:
        return lambdai

    i, j = torch.triu_indices(labels.size(0), labels.size(0), 1)

    s_i, s_j = scores[i], scores[j]
    y_i, y_j = labels[i], labels[j]

    S_ij = (y_i > y_j).float() + (y_i < y_j).float() * -1

    lambdaij = 0.5 * (1 - S_ij) - torch.sigmoid(-(s_i - s_j))

    _, indices = torch.sort(scores, descending=True)
    ranks = torch.zeros_like(scores, dtype=torch.long)
    ranks[indices] = torch.arange(scores.size(0), device=scores.device)

    gains = 2**labels - 1
    discounts = torch.log2(
        torch.arange(labels.size(0), device=scores.device, dtype=torch.float) + 2
    )

    idcg = torch.sum(
        (2 ** labels[torch.argsort(labels, descending=True)] - 1) / discounts
    )
    idcg = torch.max(idcg, torch.tensor(1e-10))

    ndcg_delta = torch.zeros_like(lambdaij)
    for k in range(len(i)):
        doc_i, doc_j = i[k].item(), j[k].item()
        rank_i, rank_j = ranks[doc_i].item(), ranks[doc_j].item()

        dcg_current = (
            gains[doc_i] / discounts[rank_i] + gains[doc_j] / discounts[rank_j]
        )

        dcg_swapped = (
            gains[doc_i] / discounts[rank_j] + gains[doc_j] / discounts[rank_i]
        )

        ndcg_delta[k] = abs(dcg_current - dcg_swapped) / idcg

    lambdaij = lambdaij * ndcg_delta
    lambdai.index_add_(0, i, lambdaij)
    lambdai.index_add_(0, j, -lambdaij)

    return lambdai.view(-1, 1)

    ## END SOLUTION


def mean_lambda_list(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean and squared mean of LambdaRank updates.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        A tensor containing the mean and squared mean lambda values.
    """
    lambda_values = listwise_loss(scores, labels)
    return torch.stack(
        [
            lambda_values.mean(),
            torch.square(lambda_values).mean(),
        ]
    )


def listNet_loss(
    output: torch.Tensor, target: torch.Tensor, grading: bool = False
) -> torch.Tensor:
    """
    Computes the ListNet loss, introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".

    This loss is based on the probability distributions of ranking scores and relevance labels.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions of shape [1, topk, 1].
    target : torch.Tensor
        Ground truth labels of shape [1, topk].
    grading : bool, optional
        If True, returns additional debugging information. Default is False.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        If `grading=False`, returns a single loss value as a tensor.
        If `grading=True`, returns a tuple containing the loss and additional debugging information.
    """
    eps = 1e-10  # Small epsilon value for numerical stability

    ## BEGIN SOLUTION
    output = output.squeeze(-1)
    target = target.view(1, -1)

    preds_smax = torch.softmax(output, dim=-1)
    true_smax = torch.softmax(target, dim=-1)
    preds_log = torch.log(torch.clamp(preds_smax, min=eps))

    # cross-entropy loss
    loss = -torch.sum(true_smax * preds_log)
    ## END SOLUTION

    if grading:
        return loss, {
            "preds_smax": preds_smax,
            "true_smax": true_smax,
            "preds_log": preds_log,
        }
    else:
        return loss


def unbiased_listNet_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    propensity: torch.Tensor,
    clipping: str = "default_clip",
    grading: bool = False,
) -> torch.Tensor:
    """
    Computes the Unbiased ListNet loss, incorporating propensity scores for unbiased learning to rank.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions of shape [1, topk, 1].
    target : torch.Tensor
        Ground truth labels of shape [1, topk].
    propensity : torch.Tensor
        Propensity scores of shape [1, topk] or [topk], used for debiasing.
    grading : bool, optional
        If True, returns additional debugging information. Default is False.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        If `grading=False`, returns a single loss value as a tensor.
        If `grading=True`, returns a tuple containing the loss and additional debugging information.
    """
    eps = 1e-10  # Small epsilon value for numerical stability

    # Clip propensity scores to avoid division by small values, improving stability and lowering variance
    # Apply different clipping strategies
    if clipping == "no_clip":
        stable_propensity = propensity  # No modification
    elif clipping == "default_clip":
        stable_propensity = propensity.clip(0.01, 1)  # Fixed range
    elif clipping == "percentile_clip":
        min_clip = torch.quantile(propensity, 0.01)  # 1st percentile
        stable_propensity = propensity.clip(min_clip, 1)
    elif clipping == "sigmoid_clip":
        stable_propensity = torch.sigmoid(
            5 * (propensity - 0.5)
        )  # Sigmoid normalization
    elif clipping == "log_clip":
        stable_propensity = torch.log(1 + propensity)  # Logarithmic scaling
    else:
        raise ValueError(f"Invalid clipping method: {clipping}")

    ## BEGIN SOLUTION
    output = output.squeeze(-1)
    target = target.view(1, -1)

    preds_smax = torch.softmax(output, dim=-1)
    true_smax = torch.softmax(target / stable_propensity, dim=-1)
    preds_log = torch.log(preds_smax + eps)

    loss = -torch.sum(true_smax * preds_log)
    ## END SOLUTION

    if grading:
        return loss, {
            "preds_smax": preds_smax,
            "true_smax": true_smax,
            "preds_log": preds_log,
        }
    else:
        return loss
