import tensorflow as tf
from keras.layers import Input
from keras.models import Model, Sequential
from tensorflow.keras import layers
import tensorflow_addons as tfa
from Classification3D.utils import *

input_shape = TARGET_SHAPE
projection_dim = 128
batch_size = 8

def get_augmentations():
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

# Encoder 3D baseado no Med3D
def create_encoder():
    inputs = Input(shape=input_shape)
    x = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=2)(x)
    x = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=2)(x)
    x = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    return Model(inputs, x, name="encoder")

# Projeção
def create_projection_head():
    inputs = Input(shape=(256,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(projection_dim)(x)
    return Model(inputs, x, name="projection_head")

# Construção do modelo SimCLR
class SimCLR(Model):
    def __init__(self, encoder, projection_head, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.temperature = temperature

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = tfa.losses.NTXentLoss(temperature=self.temperature)

    def train_step(self, data):
        images_1, images_2 = data
        with tf.GradientTape() as tape:
            features_1 = self.projection_head(self.encoder(images_1))
            features_2 = self.projection_head(self.encoder(images_2))
            loss = self.loss_fn(features_1, features_2)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

