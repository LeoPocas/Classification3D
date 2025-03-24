import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from Classification3D.utils import TARGET_SHAPE

def augment(volume):
    """Aplica aumentações leves ao volume."""
    volume = tf.image.random_flip_left_right(volume)
    volume = tf.image.random_flip_up_down(volume)
    volume = tf.image.random_brightness(volume, max_delta=0.1)
    return volume

class BYOL(tf.keras.Model):
    def __init__(self, input_shape):
        super(BYOL, self).__init__()
        self.encoder = self.build_encoder(input_shape)
        self.predictor = self.build_predictor()
    
    def build_encoder(self, input_shape):
        base_model = keras.applications.ResNet50(
            include_top=False, input_shape=input_shape, pooling='avg'
        )
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=True)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256)(x)
        return keras.Model(inputs, x, name="encoder")
    
    def build_predictor(self):
        model = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256)
        ], name="predictor")
        return model
    
    def call(self, inputs, training=True):
        view1, view2 = inputs
        proj1 = self.encoder(augment(view1), training=training)
        proj2 = self.encoder(augment(view2), training=training)
        pred1 = self.predictor(proj1)
        pred2 = self.predictor(proj2)
        return pred1, pred2

def byol_loss(pred1, pred2, target1, target2):
    loss1 = tf.losses.cosine_similarity(pred1, tf.stop_gradient(target2))
    loss2 = tf.losses.cosine_similarity(pred2, tf.stop_gradient(target1))
    return (loss1 + loss2) / 2

input_shape = TARGET_SHAPE
byol_model = BYOL(input_shape)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Exemplo de chamada de treinamento
def train_step(model, batch):
    view1, view2 = batch
    with tf.GradientTape() as tape:
        pred1, pred2 = model((view1, view2), training=True)
        loss = byol_loss(pred1, pred2, pred1, pred2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
