import theano

def align_targets(predictions, targets):
    """Helper function turning a target 1D vector into a column if needed.
    This way, combining a network of a single output unit with a target vector
    works as expected by most users, not broadcasting outputs against targets.
    Parameters
    ----------
    predictions : Theano tensor
        Expression for the predictions of a neural network.
    targets : Theano tensor
        Expression or variable for corresponding targets.
    Returns
    -------
    predictions : Theano tensor
        The predictions unchanged.
    targets : Theano tensor
        If `predictions` is a column vector and `targets` is a 1D vector,
        returns `targets` turned into a column vector. Otherwise, returns
        `targets` unchanged.
    """
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
            getattr(targets, 'ndim', None) == 1):
        targets = as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets

def binary_jaccard_index(predictions,targets):
    predictions,targets = align_targets(predictions, targets)
    intersection = theano.tensor.minimum(predictions, targets)
    union = theano.tensor.maximum(predictions, targets)
    return 1 - intersection.sum() / union.sum()