from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from unet3dModel import load_acdc_data_3d, unet_3d, weighted_dice_coefficient, weighted_categorical_crossentropy
from collections import Counter
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)]) 
    except RuntimeError as e:
        print(e)

# Carregar dados de treinamento
train_images, train_masks = load_acdc_data_3d('./ACDC/database/training/')
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.1, random_state=42)

class_counts = Counter(train_masks.flatten())  # Contagem de pixels por classe
total_pixels = sum(class_counts.values())  # Total de pixels
class_weights = {class_id: total_pixels / count for class_id, count in class_counts.items()}  # Pesos inversamente proporcionais
max_weight = max(class_weights.values())
class_weights = {k: v / max_weight for k, v in class_weights.items()}  # Normalizar pesos
print("Pesos calculados:", class_weights)

# Converter os pesos para uma lista ordenada por índice de classe
class_weights_list = [class_weights.get(i, 1.0) for i in range(len(class_weights))]

# Criar o modelo
model = unet_3d()

loss_fn = weighted_categorical_crossentropy(class_weights_list)
dice_metric = weighted_dice_coefficient(class_weights_list)  # Encapsula os pesos na métrica

model.compile(optimizer=Adam(learning_rate=1e-2),
              loss=loss_fn,
              metrics=[dice_metric])

# Callbacks para checkpoint e early stopping
callbacks = [
    ModelCheckpoint("./weights/unet.weights.keras", 
                    save_best_only=True, 
                    monitor='val_loss',  
                    verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4, min_lr=1e-5)
]
# Treinamento
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          batch_size=6,  
          callbacks=callbacks)

# Avaliação no conjunto de teste
test_images, test_masks = load_acdc_data_3d('./ACDC/database/testing/')
results = model.evaluate(test_images, test_masks, verbose=1, batch_size=4)
print(results)