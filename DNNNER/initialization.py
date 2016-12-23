import numpy as np
import tensorflow as tf


def xavier_weight_init():
    """
    Returns function that creates random tensor. 

    The specified function will take in a shape (tuple or 1-d array) and must
    return a random tensor of the specified shape and must be drawn from the
    Xavier initialization distribution.

    Hint: You might find tf.random_uniform useful.
    """
    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.

        This function will be used as a variable scope initializer.

        https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

        Args:
            shape: Tuple or 1-d array that species dimensions of requested tensor.
            kwargs: other arguments
        Returns:
            out: tf.Tensor of specified shape sampled from Xavier distribution.
        """
        m = shape[0]
        n = shape[1] if len(shape) > 1 else shape[0]

        bound = np.sqrt(6) / np.sqrt(m + n)
        out = tf.random_uniform(shape, minval=-bound, maxval=bound)
        return out
    # Returns defined initializer function.
    return _xavier_initializer
