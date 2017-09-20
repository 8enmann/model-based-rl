import numpy as np
import random

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def read_tensorboard(path_to_events_file, tag):
    """This example supposes that the events file contains summaries with a
    summary value tag 'loss'.  These could have been added by calling
    `add_summary()`, passing the output of a scalar summary op created with
    with: `tf.scalar_summary(['loss'], loss_tensor)`.
    """
    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag == tag:
                yield v.simple_value
