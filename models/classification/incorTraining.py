import tensorflow as tf
import numpy as np
import gc
from Classification3D.models.classification.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet, build_med3d_with_ssl
from Classification3D.preprocessing.load_mms import load_mms_data, load_mms_data_dual_input
from Classification3D.preprocessing.load_data import load_3d_roi_sep
from Classification3D.preprocessing.loadIncor import load_incor_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from Classification3D.utils import LABEL_MAPPING_MMS, ACDC_REESPACADO_TESTING, WEIGHT_PATH

# Configuração para usar precisão mista
mixed_precision.set_global_policy('float32')

# Configuração da GPU (opcional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16096)])
    except RuntimeError as e:
        print(e)

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        x_val_img, y_val = self.validation_data  # Dados de validação
        y_pred = self.model.predict(x_val_img, batch_size=self.batch_size)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
        print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)

images, labels = load_incor_data()

x_train_img, x_val_img, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=36
)

model = build_med3d()

# Compilar o modelo
optimizer = Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
callbacks = [
    ModelCheckpoint(WEIGHT_PATH + "incor_resnet.weights.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.93, patience=4, min_lr=1e-6),
    ConfusionMatrixCallback(validation_data=(x_val_img, y_val), batch_size=5)
]

history = model.fit(
    x_train_img,
    y_train,
    validation_data=(x_val_img, y_val),
    epochs=200, batch_size=5,
    callbacks=callbacks,
    verbose=2
)

# Testar o modelo com os dados de teste
test_images, test_labels = load_incor_data(training=False)

results_train = model.evaluate(x_val_img, y_val, verbose=0)

print("Resultados do melhor modelo no conjunto de treino:", results_train)

del x_train_img, x_val_img, y_train, y_val
gc.collect()

results = model.evaluate(test_images, test_labels, verbose=1)

print("Resultados no conjunto de teste:", results)

y_test_pred = model.predict(test_images, batch_size=6)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

gc.collect()