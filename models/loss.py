import tensorflow as tf 

def combined_loss(alpha=0.5):
    def loss_fn(y_true, y_pred):
        focal = focal_loss(gamma=2., alpha=0.25)(y_true, y_pred)
        dice = 1 - tf.reduce_mean((2 * y_true * y_pred + 1e-7) / (y_true + y_pred + 1e-7))
        return alpha * focal + (1 - alpha) * dice
    return loss_fn

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        focal_weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = -focal_weight * y_true * tf.math.log(y_pred)
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed