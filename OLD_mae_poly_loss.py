
"""
MAE POLY LOSS
"""


import tensorflow as tf


def construct(params):
    try:
        alpha = float(params["alpha"])
    except KeyError as e:
        raise KeyError("mae_poly_loss.construct(): " +
                       "alpha must be specified!") from e
    except ValueError as e:
        raise ValueError("mae_poly_loss.construct(): " +
                         "alpha must be a float: received: '%s'" %
                         str(alpha)) from e
    print("mae_poly_loss: alpha: %0.3f" % alpha)
    loss_function = mae_poly_loss(alpha)
    return loss_function

def mae_poly_loss(alpha):

    def loss(y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        second = (1 - y_true) ** alpha

        return tf.reduce_mean(mae * second)
    return loss
