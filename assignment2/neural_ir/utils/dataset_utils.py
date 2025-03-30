# TODO: implement this method
def read_pairs(path: str):
    """
    Read tab-delimited pairs from file.
    Parameters
    ----------
    path: str
        path to the input file
    Returns
    -------
        a list of pair tuple
    """
    # BEGIN SOLUTION
    result = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            result.append((parts[0], parts[1]))
    return result
    # END SOLUTION


# TODO: implement this method
def read_triplets(path: str):
    """
    Read tab-delimited triplets from file.
    Parameters
    ----------
    path: str
        path to the input file
    Returns
    -------
        a list of triplet tuple
    """
    # BEGIN SOLUTION
    result = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            result.append((parts[0], parts[1], parts[2]))
    return result
    # END SOLUTION
