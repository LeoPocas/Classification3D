import tensorflow as tf
import numpy as np
from Classification3D.models.classification.cnn3d_sep import cnn_3d_model, build_med3d
from Classification3D.preprocessing.load_data import load_4d_and_extract_3d_volumes, load_4d_roi_sep
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import mixed_precision
from Classification3D.models.loss import combined_loss
mixed_precision.set_global_policy('float32')

from ...utils import LABEL_MAPPING, ACDC_TRAINING_PATH, ACDC_TESTING_PATH, WEIGHT_PATH

# Configuração da GPU (opcional)
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

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, batch_size=self.batch_size)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch+1}:\n", cm)
        
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING.keys()))
        print(f"\nRelatório de Classificação após época {epoch+1}:\n", cr)

# Carregar os dados
images, labels = load_4d_roi_sep()

# Realizar data augmentation
#images_augmented, labels_augmented = data_augmentation_3d(images, labels)

print(f"Imagens: {images.shape}, Labels: {labels.shape}")

# Dividir os dados após o augmentation
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=33)

#model = cnn_3d_model()
model = build_med3d()

# Compilar o modelo
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=combined_loss(alpha=0.5), metrics=['accuracy'])

# Configurar os callbacks
callbacks = [
    ModelCheckpoint(WEIGHT_PATH + "med3d_4d_roi.weights.keras", save_best_only=False, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=1e-6),
    ConfusionMatrixCallback(validation_data=(x_val, y_val), batch_size=20)
]

# Treinar o modelo
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=35, batch_size=20,
    callbacks=callbacks,
    verbose=2
)

test_images, test_masks = load_4d_roi_sep(ACDC_TESTING_PATH)
results = model.evaluate(test_images, test_masks, verbose=1)
print(results)