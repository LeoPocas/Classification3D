import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, concatenate, Dropout, BatchNormalization, Conv3D, MaxPooling3D, UpSampling3D

tf.random.set_seed(42)
epsilon = 1e-5

NUM_CLASSES = 4  # Altere para o número de classes do seu problema

def load_acdc_data_3d(data_path, target_size=(256, 256, 16)):
    images, masks = [], []
    patients = os.listdir(data_path)

    for patient in patients:
        patient_path = os.path.join(data_path, patient)
        
        # Filtrar arquivos com e sem o sufixo "gt" para imagens e máscaras
        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz'):
                file_path = os.path.join(patient_path, filename)
                
                if 'gt' in filename and '4d' not in filename:
                    mask = nib.load(file_path).get_fdata()
                    mask = resize(mask, target_size, mode='constant', preserve_range=True).astype(np.int32)
                    masks.append(mask)
                elif 'frame' in filename and '4d' not in filename:
                    img = nib.load(file_path).get_fdata()
                    img = resize(img, target_size, mode='constant', preserve_range=True)
                    img = (img - np.mean(img)) / (np.std(img) + epsilon)  # Normalização
                    images.append(img)

    return np.array(images)[..., np.newaxis], np.array(masks)

def unet_3d(input_size=(256, 256, 16, 1), num_classes=NUM_CLASSES):
    inputs = Input(input_size)
    
    def conv_block_3d(x, filters):
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x
    
    conv1 = conv_block_3d(inputs, 64)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = conv_block_3d(pool1, 96)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = conv_block_3d(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = conv_block_3d(pool3, 256)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    conv5 = conv_block_3d(pool4, 512)
    
    def upsample_concat_3d(x, conv, filters):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3D(filters, (2, 2, 2), activation='relu', padding='same')(x)
        x = concatenate([x, conv], axis=-1)
        x = conv_block_3d(x, filters)
        return x
    
    conv6 = upsample_concat_3d(conv5, conv4, 256)
    conv7 = upsample_concat_3d(conv6, conv3, 128)
    conv8 = upsample_concat_3d(conv7, conv2, 64)
    conv9 = upsample_concat_3d(conv8, conv1, 32)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # Obter as previsões como probabilidades
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(weights))

        # Calcular a perda ponderada
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred) * weights, axis=-1)
        return tf.reduce_mean(loss)

    return loss_fn

def weighted_dice_coefficient(class_weights):
    def dice_fn(y_true, y_pred, smooth=1):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(class_weights))
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=len(class_weights))

        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2, 3])  # Soma por classe
        denominator = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2, 3])

        dice = (2. * intersection + smooth) / (denominator + smooth)
        weighted_dice = tf.reduce_sum(dice * tf.constant(class_weights, dtype=tf.float32)) / tf.reduce_sum(tf.constant(class_weights, dtype=tf.float32))
        return weighted_dice
    return dice_fn
