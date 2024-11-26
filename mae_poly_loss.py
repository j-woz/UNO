import tensorflow as tf
def mae_poly_loss(alpha):
    def loss(y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        second = (1-y_true)**alpha
        
        return tf.reduce_mean(mae*second)
    return loss