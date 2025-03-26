import tensorflow as tf
from keras.layers import Input, MaxPooling3D, BatchNormalization, Conv3D, GlobalAveragePooling3D, Dense, RandomFlip, RandomRotation, RandomZoom
from keras.models import Model, Sequential
from Classification3D.utils import *
from Classification3D.models.residual_block import residual_block_3d

input_shape = TARGET_SHAPE
projection_dim = 128

def nt_xent_loss(z_i, z_j, temperature=0.1):
    """
    Calcula a NT-Xent Loss para duas projeções (z_i e z_j).
    
    :param z_i: Features do primeiro conjunto (batch_size, projection_dim)
    :param z_j: Features do segundo conjunto (batch_size, projection_dim)
    :param temperature: Fator de temperatura para a similaridade
    :return: Média da perda
    """
    # Normaliza os vetores para magnitude 1
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Concatena os embeddings
    representations = tf.concat([z_i, z_j], axis=0)  # (2 * batch_size, projection_dim)

    # Calcula o produto escalar (similaridade) entre todos os pares
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)  # (2 * batch_size, 2 * batch_size)

    # Máscara para remover a similaridade consigo mesmo
    batch_size = tf.shape(z_i)[0]
    mask = tf.linalg.diag(tf.ones(2 * batch_size))  # (2 * batch_size, 2 * batch_size)

    # Escala com o fator de temperatura
    similarity_matrix = similarity_matrix / temperature

    # Labels para o batch (pares positivos)
    labels = tf.one_hot(tf.range(batch_size), 2 * batch_size)
    labels = tf.concat([labels, labels], axis=0)  # (2 * batch_size, 2 * batch_size)

    # Calcula a perda cruzada entre os logits e os labels
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, similarity_matrix * (1.0 - mask))
    return tf.reduce_mean(loss)

def get_augmentations():
    return Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

# Encoder 3D baseado no Med3D
def create_encoder_cnn():
    inputs = Input(shape=(*input_shape, 1), name='image_input')
    x = Conv3D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = Conv3D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = Conv3D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling3D()(x)
    return Model(inputs, x, name="encoder")

def create_encoder_resnet():
    inputs = Input(shape=(*input_shape, 1), name='image_input')
    x = Conv3D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = residual_block_3d(x, filters=64)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = residual_block_3d(x, filters=128)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = residual_block_3d(x, filters=256)
    x = GlobalAveragePooling3D()(x)
    
    return Model(inputs, x, name="encoder")

# Projeção
def create_projection_head():
    inputs = Input(shape=(256,))
    x = Dense(256, activation="relu")(inputs)
    x = Dense(projection_dim)(x)
    return Model(inputs, x, name="projection_head")

# Construção do modelo SimCLR
class SimCLR(tf.keras.Model):
    def __init__(self, encoder, projection_head, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.temperature = temperature

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = lambda z_i, z_j: nt_xent_loss(z_i, z_j, self.temperature)  # Usando a função customizada

    def train_step(self, data):
        # Recebe os pares de imagens aumentadas
        images_1, images_2 = data
        with tf.GradientTape() as tape:
            # Extrai as features das imagens
            features_1 = self.projection_head(self.encoder(images_1))
            features_2 = self.projection_head(self.encoder(images_2))

            # Calcula a perda
            loss = self.loss_fn(features_1, features_2)

        # Atualiza os pesos do modelo
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}


