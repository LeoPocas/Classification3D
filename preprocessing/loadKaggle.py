import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import random

def load_nifti(file_path):
    """Carrega um volume NIfTI e o converte para um tensor normalizado."""
    img = nib.load(file_path)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.expand_dims(data, axis=-1)  # Adiciona canal para compatibilidade com CNNs
    return data.astype(np.float32)

def data_augmentation(volume):
    """Aplica transformações aleatórias ao volume para SimCLR."""
    # Flip horizontal e vertical
    if random.random() > 0.5:
        volume = tf.image.flip_left_right(volume)
    if random.random() > 0.5:
        volume = tf.image.flip_up_down(volume)
    
    # Mudança de brilho
    volume = tf.image.random_brightness(volume, max_delta=0.1)
    
    # Mudança de contraste
    volume = tf.image.random_contrast(volume, lower=0.8, upper=1.2)
    
    return volume

def prepare_ssl_dataset(kaggle_dataset_path, batch_size=8):
    file_paths = [os.path.join(kaggle_dataset_path, f) for f in os.listdir(kaggle_dataset_path) if f.endswith('.nii')]
    
    def generator():
        for file_path in file_paths:
            volume = load_nifti(file_path)
            aug_1 = data_augmentation(volume)
            aug_2 = data_augmentation(volume)
            yield aug_1, aug_2
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32)
    ))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
