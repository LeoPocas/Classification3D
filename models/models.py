import numpy as np
from keras.layers import Input, Conv3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, MaxPooling3D, BatchNormalization, Dropout, add, concatenate, LeakyReLU, Reshape, Multiply, ReLU, Flatten
from keras.models import Model
from keras.regularizers import l2
from scipy.ndimage import zoom
from Classification3D.models.ssl.simCLR import create_encoder_cnn, create_encoder_resnet
from Classification3D.models.residual_block import residual_block, residual_block_3d
from Classification3D.utils import WEIGHT_PATH, TARGET_SHAPE, NUM_CLASSES, NUM_CLASSES_MMS

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

def build_med3d(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES_MMS, patiente_analysis=False): 
    #ResNet: Residual Network, ela ajuda a não termos perda elevada no gradiente utilizando skip connections
    if patiente_analysis:
        inputs = Input(shape=(*input_shape, 1), name='image_input')
        metadata_input = Input(shape=(3,), name='metadata_input') 
    else: 
        inputs = Input(shape=(*input_shape, 1))

    # Initial Convolution
    x = Conv3D(64, kernel_size=1, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=2, padding='same')(x)

    # Residual Blocks
    # x = residual_block_3d(x, 32)
    x = residual_block_3d(x, 64)
    x = residual_block_3d(x, 128)
    x = residual_block_3d(x, 256)
    #x = residual_block_3d(x, 512)
    #x = residual_block_3d(x, 1024)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    # x = Conv3D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Flatten()(x)

    # metadata_x = Dense(8, activation='relu')(metadata_input)  
    # combined = concatenate([x, metadata_x])

    # Fully Connected Layers
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    if patiente_analysis:
        model = Model(inputs={'image_input': inputs, 'metadata_input': metadata_input}, outputs=outputs)
    else:
        model = Model(inputs=inputs, outputs=outputs)
    return model

def newModel(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=(*input_shape, 1))
    
    x = Conv3D(16, kernel_size=1, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = Conv3D(32, kernel_size=1, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=2, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def dualInput_Resnet(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    # Primeiro input: Volume na sístole
    systole_input = Input(shape=(*input_shape, 1), name='systole_input')
    x1 = Conv3D(64, kernel_size=1, padding='same', activation='relu')(systole_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling3D(pool_size=2, padding='same')(x1)
    x1 = residual_block_3d(x1, 64)
    x1 = residual_block_3d(x1, 128)
    x1 = residual_block_3d(x1, 256)
    x1 = GlobalAveragePooling3D()(x1)
    x1 = Flatten()(x1)

    # Segundo input: Volume na diástole
    diastole_input = Input(shape=(*input_shape, 1), name='diastole_input')
    x2 = Conv3D(64, kernel_size=1, padding='same', activation='relu')(diastole_input)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling3D(pool_size=2, padding='same')(x2)
    x2 = residual_block_3d(x2, 64)
    x2 = residual_block_3d(x2, 128)
    x2 = residual_block_3d(x2, 256)
    x2 = GlobalAveragePooling3D()(x2)
    x2 = Flatten()(x2)

    # metadata_input = Input(shape=(3,), name='metadata_input')
    # metadata_x = Dense(9, activation='relu')(metadata_input)

    # Combinação das três entradas
    combined = concatenate([x1, x2])

    # Camadas densas finais
    x = Dense(256, activation='relu')(combined)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Modelo final
    model = Model(inputs={'systole_input': systole_input, 'diastole_input': diastole_input}, 
                  outputs=outputs)
    return model

def build_med3d_with_ssl(encoder=create_encoder_resnet(),input_shape=TARGET_SHAPE, num_classes=4, encoder_weights=WEIGHT_PATH+"encoder_ssl.weights.h5", trainable=False):
    # Carrega o encoder pré-treinado
    encoder.load_weights(encoder_weights)  # Carrega os pesos treinados

    # for layer in encoder.layers[:-1]:  # Exclui a última camada
    #     layer.trainable = False

    # Descongela apenas a última camada
    encoder.layers[-1].trainable = True
    encoder.layers[-2].trainable = True
    encoder.layers[-3].trainable = True

    # Entrada para as imagens
    inputs = Input(shape=(*input_shape, 1), name='image_input')
    x = encoder(inputs)  # Passa os dados pelo encoder pré-treinado

    # Adiciona camadas do modelo supervisionado após o encoder
    metadata_input = Input(shape=(3,), name='metadata_input')  # Entrada para metadados (opcional)

    # Camadas finais para classificação
    x = Dense(256, activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Criação do modelo com as entradas e saídas
    model = Model(inputs={'image_input': inputs, 'metadata_input': metadata_input}, outputs=outputs)
    return model
