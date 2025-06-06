import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import mixed_precision
from cnn4d import build_med3d_lstm, load_acdc_data_4d
from ...utils import TARGET_SHAPE, NUM_CLASSES, LABEL_MAPPING, MAX_TIME_DIM, ACDC_TRAINING_PATH, ACDC_TESTING_PATH

mixed_precision.set_global_policy('float32')

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

def combined_loss(alpha=0.5):
    def loss_fn(y_true, y_pred):
        focal = focal_loss(gamma=2., alpha=0.25)(y_true, y_pred)
        dice = 1 - tf.reduce_mean((2 * y_true * y_pred + 1e-7) / (y_true + y_pred + 1e-7))
        return alpha * focal + (1 - alpha) * dice
    return loss_fn

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        focal_weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = -focal_weight * y_true * tf.math.log(y_pred)
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

# Carregar os dados
images, labels = load_acdc_data_4d()

print(f"Imagens: {images.shape}, Labels: {labels.shape}")

# Dividir os dados após o augmentation
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=35)

#model = cnn_3d_model()
model = build_med3d_lstm()

# Compilar o modelo
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=combined_loss(alpha=0.5), metrics=['accuracy'])

# Configurar os callbacks
callbacks = [
    ModelCheckpoint("../../weights/med4d_ltsm.weights.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=1e-6),
    ConfusionMatrixCallback(validation_data=(x_val, y_val), batch_size=2)
]

# Treinar o modelo
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=40, batch_size=2,
    callbacks=callbacks,
    verbose=2
)

test_images, test_masks = load_acdc_data_4d(ACDC_TESTING_PATH)
results = model.evaluate(test_images, test_masks, verbose=1)
print(results)
    