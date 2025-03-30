import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ltr.dataset import ClickLTRData, LTRData, QueryGroupedLTRData, qg_collate_fn
from ltr.eval import evaluate_model
from ltr.logging_policy import LoggingPolicy
from ltr.loss import (
    compute_lambda_i,
    listNet_loss,
    listwise_loss,
    pairwise_loss,
    pointwise_loss,
    unbiased_listNet_loss,
)


def train_batch(net, x, y, loss_fn, optimizer):
    """
    Performs a single training batch update for the given neural network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to be trained.
    x : torch.Tensor
        Input tensor containing the training data.
    y : torch.Tensor
        Target tensor containing the ground truth labels.
    loss_fn : callable
        Loss function used to compute the training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating the model parameters.

    Returns
    -------
    None
    """
    optimizer.zero_grad()
    out = net(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()


# TODO: Implement this!
def train_pointwise(net, params, data):
    """
    This function should train a Pointwise network.

    The network is trained using the Adam optimizer


    Note: Do not change the function definition!


    Hints:
    1. Use the LTRData class defined above
    2. Do not forget to use net.train() and net.eval()

    Inputs:
            net: the neural network to be trained

            params: params is an object which contains config used in training
                (eg. params.epochs - the number of epochs to train).
                For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models.

    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pointwise_loss

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        LTRData(data, "train"), batch_size=params.batch_size, shuffle=True
    )

    # Step 2: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()

        # Step 3: Iterate over each batch of data and use the train_batch function to train the model
        for x, y in tqdm(train_dl, desc=f"Training Epoch {epoch+1}"):
            train_batch(net, x, y, loss_fn, optimizer)

        # Step 4: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        with torch.no_grad():
            train_metrics = evaluate_model(data, net, "train")
            val_metrics = evaluate_model(data, net, "validation")

        # Step 5: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(train_metrics)
        val_metrics_epoch.append(val_metrics)

    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_batch_vector(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, NUM_FEATURES), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.
    The loss function returns a vector of size [N, 1], the same as the output of network.

    Input:  x: feature matrix, a [N, NUM_FEATURES] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    ### BEGIN SOLUTION
    # Step tips:
    # Step 1: Zero the gradients of the optimizer
    optimizer.zero_grad()
    # Step 2: Forward pass the input through the network using the net
    out = net(x).view(-1)
    # Step 3: Compute the loss using the loss_fn
    loss = loss_fn(out, y.view(-1))
    # Step 4: Backward pass to compute the gradients
    if loss is None:
        return
    loss.backward()
    # Step 5: Update the weights using the optimizer
    optimizer.step()
    ### END SOLUTION

# TODO: Implement this!
def train_pairwise(net, params, data):
    """
    This function should train the given network using the pairwise loss

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1

    Hint: Consider the case when the loss function returns 'None'

    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.
    """

    val_metrics_epoch = []
    train_metrics_epoch = []

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        LTRData(data, "train"), batch_size=params.batch_size, shuffle=True
    )
    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)
    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()
        # Step 4: Iterate over each batch of data and use the train_batch_vector function to train the model
        for x, y in tqdm(train_dl, desc=f"Training Epoch {epoch+1}"):
            if y.size(0) < 2:
                continue
            # Step 5: Compute the pairwise loss using the pairwise_loss function
            # Step 6: Compute the gradients and update the weights using the optimizer
            train_batch_vector(net, x, y, pairwise_loss, optimizer)
        # Step 7: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        with torch.no_grad():
            train_metrics = evaluate_model(data, net, "train")
            val_metrics = evaluate_model(data, net, "validation")
        # Step 8: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(train_metrics)
        val_metrics_epoch.append(val_metrics)
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_pairwise_spedup(net, params, data):
    """
    This function should train the given network using the sped-up pairwise loss.

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1.

    Parameters
    ----------
    net : The neural network to be trained
    params : Object containing config used in training (e.g., params.epochs for number of epochs)
    data : Training data.

    Returns
    -------
    dict
        Contains "metrics_val" (a list of dictionaries) and "metrics_train" (a list of dictionaries).
        "metrics_val" contains metrics computed after each epoch on the validation set.
    """
    val_metrics_epoch = []
    train_metrics_epoch = []

    ## BEGIN SOLUTION
    train_loader = DataLoader(QueryGroupedLTRData(data, "train"), batch_size=params.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    for epoch in range(params.epochs):
        net.train()
        for docs, labels in train_loader:
            optimizer.zero_grad()
            out = net(docs)  
            scores = out.squeeze(0)  
            lambda_i = compute_lambda_i(scores, labels.squeeze(0)).unsqueeze(0) 
            out.backward(lambda_i)
            optimizer.step()

        net.eval()
        train_metrics = evaluate_model(data, net, "train")
        val_metrics = evaluate_model(data, net, "validation")
        train_metrics_epoch.append(train_metrics)
        val_metrics_epoch.append(val_metrics)

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}
    ## END SOLUTION



# TODO: Implement this!
def train_listwise(net, params, data):
    """
    This function should train the given network using the listwise (LambdaRank) loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        QueryGroupedLTRData(data, "train"), batch_size=params.batch_size, shuffle=True
    )
    # Step 2: Create your Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()
        # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
        for x, y in tqdm(train_dl, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            scores = net(x).view(-1)
            # Step 5: Compute the lambda gradient values for the listwise (LambdaRank) loss with the compute_lambda_i method on the scores and the output labels
            lambda_i = compute_lambda_i(scores, y.view(-1)).view(scores.shape) 
            # Step 6: Bacward from the scores with the use of the lambda gradient values
            scores.backward(lambda_i)
            # Step 7: Update the weights using the optimizer
            optimizer.step()
        # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        with torch.no_grad():
            train_metrics = evaluate_model(data, net, "train")
            val_metrics = evaluate_model(data, net, "validation")
        # Step 9: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(train_metrics)
        val_metrics_epoch.append(val_metrics)

    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def logit_to_prob(logit):
    """
    Converts logits to probabilities using the sigmoid function.

    Parameters
    ----------
    logit : torch.Tensor
        Input tensor containing logit values.

    Returns
    -------
    torch.Tensor
        Output tensor containing probabilities in the range [0,1].
    """

    ### BEGIN SOLUTION
    return torch.sigmoid(logit)
    ### END SOLUTION


# TODO: Implement this!
def train_biased_listNet(net, params, data):
    """
    This function should train the given network using the (biased) listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """

    val_metrics_epoch = []

    assert params.batch_size == 1

    logging_policy = LoggingPolicy()

    ### BEGIN SOLUTION
    # Step 1: Create the train data loader
    # Step 2: Create the validation data loader
    # Step 3: Create the Adam optimizer
    # Step 4: Iterate over the epochs and data entries
    # Step 5: Train the model using the listNet loss
    # Step 6: Evaluate on the validation set every epoch
    # Step 7: Store the metrics in val_metrics_epoch

    train_dataset = ClickLTRData(data, logging_policy)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    if hasattr(net, "init_parameters"):
        net.init_parameters()

    optimizer = Adam(net.parameters(), lr=params.lr)
    for _ in tqdm(range(params.epochs)):
        net.train()
        for features, clicks, positions in train_loader:
            out = net(features)
            loss = listNet_loss(out, clicks)
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)

    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch}


# TODO: Implement this!
def train_unbiased_listNet(net, params, data):
    """
    This function should train the given network using the unbiased_listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1
    Note: For this function, params should also have the propensity attribute


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)
            params.propensity - the propensity values used for IPS in unbiased_listNet

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """

    val_metrics_epoch = []

    assert params.batch_size == 1
    assert hasattr(params, "propensity")

    # LOGGING POLICY
    if hasattr(params, "log_policy_noise"):
        print(f"[*] Logging policy noise: {params.log_policy_noise}")
        noise = params.log_policy_noise
    else:
        noise = 0.05
    if hasattr(params, "log_policy_topk"):
        print(f"[*] Logging policy topk: {params.log_policy_topk}")
        topk = params.log_policy_topk
    else:
        topk = 20

    if hasattr(params, "gaussian_noise"):
        print(f"[*] Logging policy gaussian noise: {params.gaussian_noise}")
        gaussian_noise = params.gaussian_noise
        gaussian_noise_std = params.gaussian_noise_std
    else:
        gaussian_noise = False
        gaussian_noise_std = 0.05

    logging_policy = LoggingPolicy(
        noise=noise,
        topk=topk,
        gaussian_noise=gaussian_noise,
        gaussian_noise_std=gaussian_noise_std,
    )

    # CLIPPING
    if hasattr(params, "clip"):
        print(f"[*] Clipping: {params.clip}")
        clip = params.clip
    else:
        clip = "default_clip"
        print(f"[*] Clipping: {clip}")

    ### BEGIN SOLUTION
    # Step 1: Create the train data loader
    # Step 2: Create the validation data loader
    # Step 3: Create the Adam optimizer
    # Step 4: Iterate over the epochs and data entries
    # Step 5: Train the model using the unbiased listNet loss
    # Step 6: Evaluate on the validation set every epoch
    # Step 7: Store the metrics in val_metrics_epoch

    train_data = DataLoader(
        ClickLTRData(data, logging_policy), batch_size=params.batch_size, shuffle=True
    )
    if hasattr(net, "init_parameters"):
        net.init_parameters()
    optimizer = Adam(net.parameters(), lr=params.lr)

    for _ in tqdm(range(params.epochs)):
        net.train()
        for features, clicks, positions in train_data:
            scores = net(features)
            loss = unbiased_listNet_loss(
                scores, clicks, params.propensity[positions.long()], clip
            )

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_metrics_epoch.append(evaluate_model(data, net, "validation"))
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch}
