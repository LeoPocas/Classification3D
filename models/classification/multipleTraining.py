import tensorflow as tf
import numpy as np
import gc
from Classification3D.models.classification.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet, build_med3d_with_ssl
from Classification3D.preprocessing.load_mms import load_mms_data, load_mms_data_dual_input
from Classification3D.preprocessing.load_data import load_3d_roi_sep
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

# Carregar os dados do M&Ms
imagesMms, labelsMms, patient_dataMms = load_mms_data()
imagesAcdc, labelsAcdc, patient_dataAcdc = load_3d_roi_sep()

# images = np.concatenate([imagesMms, imagesAcdc], axis=0)
# labels = np.concatenate([labelsMms, labelsAcdc], axis=0)
# x_train_img, x_val_img, y_train, y_val = train_test_split(
#     images, labels, test_size=0.2, random_state=36
# )

x_train_mms, x_val_mms, y_train_mms, y_val_mms = train_test_split(
    imagesMms, labelsMms, test_size=0.2, random_state=36)
x_train_acdc, x_val_acdc, y_train_acdc, y_val_acdc = train_test_split(
    imagesAcdc, labelsAcdc, test_size=0.2, random_state=36)

x_train_img = np.concatenate([x_train_mms, x_train_acdc], axis=0)
y_train = np.concatenate([y_train_mms, y_train_acdc], axis=0)
x_val_img = np.concatenate([x_val_mms, x_val_acdc], axis=0)
y_val = np.concatenate([y_val_mms, y_val_acdc], axis=0)

model = build_med3d()

# Compilar o modelo
optimizer = Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
callbacks = [
    ModelCheckpoint(WEIGHT_PATH + "multiple_resnet.weights.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=6, min_lr=5e-6),
    ConfusionMatrixCallback(validation_data=(x_val_img, y_val), batch_size=6)
]

history = model.fit(
    x_train_img,
    y_train,
    validation_data=(x_val_img, y_val),
    epochs=500, batch_size=6,
    callbacks=callbacks,
    verbose=2
)

# Testar o modelo com os dados de teste
del x_train_img, x_val_img, y_train, y_val
gc.collect()

test_imagesMms, test_labelsMms, test_patient_dataMms = load_mms_data(training=False)
test_imagesAcdc, test_labelsAcdc, test_patient_dataAcdc = load_3d_roi_sep(ACDC_REESPACADO_TESTING)

test_images = np.concatenate([test_imagesMms, test_imagesAcdc], axis=0)
test_labels = np.concatenate([test_labelsMms, test_labelsAcdc], axis=0)

results = model.evaluate(test_images, test_labels, verbose=1)
resultsMms = model.evaluate(test_imagesMms, test_labelsMms, verbose=1)
resultsAcdc = model.evaluate(test_imagesAcdc, test_labelsAcdc, verbose=1)

print("Resultados no conjunto de teste:", results)
print("Resultados no subconjunto do MMs de teste:", resultsMms)
print("Resultados no subconjunto do ACDC de teste:", resultsAcdc)

y_test_pred = model.predict(test_images, batch_size=6)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predMms = model.predict(test_imagesMms, batch_size=6)
y_test_pred_classesMms = np.argmax(y_test_predMms, axis=1)
y_test_trueMms = np.argmax(test_labelsMms, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueMms, y_test_pred_classesMms)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueMms, y_test_pred_classesMms, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predAcdc = model.predict(test_imagesAcdc, batch_size=6)
y_test_pred_classesAcdc = np.argmax(y_test_predAcdc, axis=1)
y_test_trueAcdc = np.argmax(test_labelsAcdc, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueAcdc, y_test_pred_classesAcdc)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueAcdc, y_test_pred_classesAcdc, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)
gc.collect()
