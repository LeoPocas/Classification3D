import os
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from keras.layers import Input, Conv3D, GlobalAveragePooling3D, Dense, MaxPooling3D, BatchNormalization, Dropout, add, ReLU, LSTM, Layer, Masking, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
TARGET_SHAPE = (128, 128, 16)
MAX_TIME_DIM = 16  # Limite máximo para a dimensão do tempo
NUM_CLASSES = 5
dataset_path = './ACDC/database/training/'


class ResidualBlock3D(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock3D, self).__init__(**kwargs)
        self.conv1 = Conv3D(filters, kernel_size, padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv3D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.shortcut_conv = Conv3D(filters, kernel_size=1, padding='same')
        self.shortcut_bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut)
        x = add([x, shortcut])
        return self.relu(x)
    
    def compute_output_shape(self, input_shape):
        return input_shape


def build_med3d_lstm(input_shape=TARGET_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=(MAX_TIME_DIM, *input_shape, 3))

    x = TimeDistributed(Conv3D(64, kernel_size=7, strides=2, padding='same', activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling3D(pool_size=3, strides=2, padding='same'))(x)

    x = TimeDistributed(ResidualBlock3D(64))(x)
    x = TimeDistributed(ResidualBlock3D(128))(x)
    x = TimeDistributed(ResidualBlock3D(256))(x)
    x = TimeDistributed(ResidualBlock3D(512))(x)

    x = TimeDistributed(GlobalAveragePooling3D())(x)

    x = Masking(mask_value=0.0)(x)  # Ignorar o padding
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

#images, labels = load_acdc_data_4d()

# Dividir dados em treino e validação
#X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Construir e compilar o modelo
#model = build_med3d_lstm(TARGET_SHAPE, NUM_CLASSES)
#model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
#model.fit(X_train, y_train, batch_size=2, epochs=40, validation_data=(X_val, y_val))
