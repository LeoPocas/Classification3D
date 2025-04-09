from keras.layers import Conv3D, BatchNormalization, Dropout, add, ReLU
from keras.regularizers import l2

def residual_block_3d(input_tensor, filters, kernel_size=2):
    x = Conv3D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv3D(filters, kernel_size=1, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = ReLU()(x)
    x = Dropout(0.05)(x)
    return x


def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv3D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU(x)
    x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    if strides > 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv3D(filters, 1, padding='same', strides=strides, kernel_regularizer=l2(1e-4))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = ReLU(x)
    return x
