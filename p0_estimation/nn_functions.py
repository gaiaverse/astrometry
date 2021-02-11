import tensorflow as tf, numpy as np



# Activation function for Gaussian Parameters
def gaussian_layer(x):
    """
    Lambda function for generating gaussian parameters
    m, s from a Dense(2) output.
    Assumes tensorflow 2 backend.

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(gaussian_layer)(outputs)

    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer

    Returns
    -------
    out_tensor : tf.Tensor

    """

    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    m, s = tf.unstack(x, num=2, axis=-1)

    # If relu input
    # m = tf.math.log(m+1e-10)

    # Add one dimension to make the right shape
    m = tf.expand_dims(m, -1)
    s = tf.expand_dims(s, -1)

    # Apply an exponential activation to bound between 0 and inf
    s = tf.keras.activations.exponential(s)

    # Join back together again
    out_tensor = tf.concat((m, s), axis=num_dims-1)

    return out_tensor

def negative_gaussian_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    # Separate the parameters
    m, s = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    m = tf.expand_dims(m, -1)
    s = tf.expand_dims(s, -1)

    # Calculate the negative log likelihood
    nll = ( tf.math.log(s) + 0.5*(y_true-m)**2 / s**2 )

    return nll# + 100*tf.math.log(s)

def logit(x, xmin, xmax):
    """
    logit function to transform variable from [xmin,xmax] to [-inf,inf]
    """
    return np.log((x-xmin)/(xmax-x))
def expit(x, xmin, xmax):
    """
    expit function to transform variable from [-inf,inf] to [xmin,xmax]
    """
    return (xmin + xmax*np.exp(x))/(1+np.exp(x))
