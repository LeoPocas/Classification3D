from Classification3D.models.ssl.simCLR import create_encoder, create_projection_head, SimCLR, get_augmentations
from Classification3D.preprocessing.loadKaggle import load_kaggle_data
from Classification3D.utils import OUTPUT_PATH, WEIGHT_PATH
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf

# Cria pares de imagens aumentadas
def prepare_simclr_data(volumes, augmentations):
    augmented_pairs = []
    for volume in volumes:
        aug_1 = augmentations(volume) 
        aug_2 = augmentations(volume)
        augmented_pairs.append((aug_1, aug_2))
    return tf.data.Dataset.from_generator(
        lambda: augmented_pairs,
        output_signature=(tf.TensorSpec(shape=volumes[0].shape, dtype=tf.float32),
                          tf.TensorSpec(shape=volumes[0].shape, dtype=tf.float32))
    )

# Instancia o encoder e o modelo SimCLR
encoder = create_encoder()
projection_head = create_projection_head()
model = SimCLR(encoder, projection_head)

# Compila o modelo
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer)

reduce_lr = ReduceLROnPlateau(
    monitor='loss',            # Métrica a ser monitorada (pode ser 'val_loss' para validação)
    factor=0.9,                # Fator de redução do learning rate (exemplo: reduz pela metade)
    patience=4,                # Número de épocas sem melhoria antes de reduzir o learning rate
    min_lr=1e-6,               # Limite inferior para o learning rate
    verbose=1                  # Exibe mensagens no console
)


# Carrega os volumes do Kaggle
volumes = load_kaggle_data()

# Divide os volumes em treino e validação
train_volumes, val_volumes = train_test_split(volumes, test_size=0.2, random_state=42)

augmentations = get_augmentations()  # Usa a função de augmentations
train_data = prepare_simclr_data(train_volumes, augmentations)

# Treina o SimCLR
model.fit(
    train_data.batch(16),
    epochs=60,
    callbacks=[reduce_lr]      # Adiciona o callback aqui
)

# Salva os pesos do encoder após o treinamento SSL
encoder.save_weights(WEIGHT_PATH + "encoder_ssl.weights.h5")
