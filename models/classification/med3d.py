import numpy as np
from keras.layers import Input, Conv3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, MaxPooling3D, BatchNormalization, Dropout, add, concatenate, LeakyReLU, Reshape, Multiply, ReLU, Flatten
from keras.models import Model, Sequential
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.activations import relu
from scipy.ndimage import zoom
import tensorflow as tf

from ...utils import TARGET_SHAPE, NUM_CLASSES

def resize_volume(volume, target_shape):
    scale_factors = [target_dim / input_dim for target_dim, input_dim in zip(target_shape, volume.shape)]
    volume_resized = zoom(volume, zoom=scale_factors, order=3)
    return volume_resized

def pad_or_crop_volume(volume, target_shape):
    current_shape = volume.shape
    padding = [(max(0, (t - c) // 2), max(0, (t - c + 1) // 2)) for t, c in zip(target_shape, current_shape)]
    cropping = [(max(0, (c - t) // 2), max(0, (c - t + 1) // 2)) for t, c in zip(target_shape, current_shape)]
    volume_padded = np.pad(volume, padding, mode='constant', constant_values=0)
    slices = [slice(c[0], -c[1] if c[1] > 0 else None) for c in cropping]
    volume_cropped = volume_padded[slices[0], slices[1], slices[2]]
    return volume_cropped

def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv3D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    if strides > 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv3D(filters, 1, padding='same', strides=strides, kernel_regularizer=l2(1e-4))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = relu(x)
    return x

def attention_block(input_tensor):
    se = GlobalAveragePooling3D()(input_tensor)
    se = Reshape((1, 1, 1, input_tensor.shape[-1]))(se)
    se = Dense(input_tensor.shape[-1] // 16, activation='relu')(se)
    se = Dense(input_tensor.shape[-1], activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

def cnn_3d_model(target_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    input_layer = Input(shape=(*target_shape, 1))
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_regularizer=l2(1e-4))(input_layer)
    x = attention_block(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = Dropout(0.4)(x)

    x1 = residual_block(x, 64)
    x1 = MaxPooling3D(pool_size=(2, 2, 1))(x1)
    x1 = Dropout(0.3)(x1)

    x2 = residual_block(x1, 128)
    x2 = MaxPooling3D(pool_size=(2, 2, 1))(x2)
    x2 = Dropout(0.2)(x2)

    x3 = residual_block(x2, 256)
    x3 = MaxPooling3D(pool_size=(2, 2, 1))(x3)
    x3 = Dropout(0.2)(x3)

    x4 = residual_block(x3, 512)
    x4 = MaxPooling3D(pool_size=(2, 2, 1))(x4)
    x4 = Dropout(0.2)(x4)

    x5 = residual_block(x4, 1024)
    x5 = MaxPooling3D(pool_size=(3, 3, 1))(x5)
    x5 = Dropout(0.2)(x5)

    gap = GlobalAveragePooling3D()(x5)
    gmp = GlobalMaxPooling3D()(x5)
    x = concatenate([gap, gmp])

    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    
    return model

def residual_block_3d(input_tensor, filters, kernel_size=7):
    x = Conv3D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv3D(filters, kernel_size=1, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    return x

def build_med3d(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=(*input_shape, 1))

    # Initial Convolution
    x = Conv3D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    # Residual Blocks
    #x = residual_block_3d(x, 32)
    #x = residual_block_3d(x, 64)
    x = residual_block_3d(x, 128)
    x = residual_block_3d(x, 256)
    #x = residual_block_3d(x, 512)
    #x = residual_block_3d(x, 1024)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Fully Connected Layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model

def newModel(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=(*input_shape, 1))
    
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = Dropout(0.25)(x)
    
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = Dropout(0.25)(x)
    
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model