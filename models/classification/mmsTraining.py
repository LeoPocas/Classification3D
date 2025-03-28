import tensorflow as tf
import numpy as np
import gc
from Classification3D.models.classification.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet, build_med3d_with_ssl
from Classification3D.preprocessing.load_mms import load_mms_data, load_mms_data_dual_input, load_mms_data_pure
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from Classification3D.utils import LABEL_MAPPING_MMS, MMs_PATH, WEIGHT_PATH

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
        x_val_img, x_val_meta, y_val = self.validation_data  # Dados de validação
        y_pred = self.model.predict(
            {'image_input': x_val_img, 'metadata_input': x_val_meta},
            batch_size=self.batch_size
        )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
        print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)

# class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
#     def __init__(self, validation_data, batch_size):
#         super().__init__()
#         self.validation_data = validation_data
#         self.batch_size = batch_size

#     def on_epoch_end(self, epoch, logs=None):
#         # Ajuste para lidar com múltiplos inputs no dataset de validação
#         x_val_systole, x_val_diastole, x_val_meta, y_val = self.validation_data  # Corrigido para quatro entradas
#         y_pred = self.model.predict(
#             {'systole_input': x_val_systole, 'diastole_input': x_val_diastole, 'metadata_input': x_val_meta},
#             batch_size=self.batch_size
#         )
#         y_pred_classes = np.argmax(y_pred, axis=1)
#         y_true = np.argmax(y_val, axis=1)
        
#         cm = confusion_matrix(y_true, y_pred_classes)
#         print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
#         cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
#         print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)


# Carregar os dados do M&Ms
images, labels, patient_data = load_mms_data()
# data, labels = load_mms_data_dual_input()
# systole_images = data['systole']
# diastole_images = data['diastole']
# patient_data = data['metadata']

# Normalizar os metadados (peso, sexo, idade)
scaler = StandardScaler()
patient_data = scaler.fit_transform(patient_data)

x_train_img, x_val_img, y_train, y_val, x_train_meta, x_val_meta = train_test_split(
    images, labels, patient_data, test_size=0.2, random_state=36
)

# x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val, x_train_meta, x_val_meta = train_test_split(
#     systole_images, diastole_images, labels, patient_data, test_size=0.1, random_state=36
# )

# model = dualInput_Resnet(num_classes=len(LABEL_MAPPING_MMS))  # Adaptado para múltiplos inputs (imagem + metadados)
# model = build_med3d()
model = build_med3d_with_ssl()

# Compilar o modelo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
callbacks = [
    ModelCheckpoint(WEIGHT_PATH + "mms_resnet.s.weights.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=6, min_lr=1e-6),
    ConfusionMatrixCallback(validation_data=(x_val_img, x_val_meta, y_val), batch_size=6)
]

# callbacks = [
#     ModelCheckpoint(WEIGHT_PATH + "med3d_dual_input.weights.keras", save_best_only=True, monitor="val_loss"),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=5e-6),
#     ConfusionMatrixCallback(
#         validation_data=(x_val_systole, x_val_diastole, x_val_meta, y_val),  # Adicionar todos os inputs e labels
#         batch_size=8
#     )
# ]

history = model.fit(
    {'image_input': x_train_img, 'metadata_input': x_train_meta},
    y_train,
    validation_data=({'image_input': x_val_img, 'metadata_input': x_val_meta}, y_val),
    epochs=120, batch_size=6,
    callbacks=callbacks,
    verbose=2
)

# history = model.fit(
#     {'systole_input': x_train_systole, 'diastole_input': x_train_diastole, 'metadata_input': x_train_meta}, 
#     y_train,
#     validation_data=(
#         {'systole_input': x_val_systole, 'diastole_input': x_val_diastole, 'metadata_input': x_val_meta}, y_val), 
#     epochs=1, batch_size=8,
#     callbacks=callbacks,
#     verbose=2
# )

# Testar o modelo com os dados de teste
del images, labels, patient_data, x_train_img, x_train_meta, x_val_img ,x_val_meta, y_train, y_val
gc.collect()

test_images, test_labels, test_patient_data = load_mms_data(training=False)
# test_data, test_labels = load_mms_data_dual_input(training = False) 

# test_patient_data = scaler.transform(test_patient_data)
# test_systole = test_data['systole']
# test_diastole = test_data['diastole']
# test_patient_data = test_data['metadata']

results = model.evaluate({'image_input': test_images, 'metadata_input': test_patient_data}, test_labels, verbose=1)
# results = model.evaluate(
#     {'systole_input': test_systole, 'diastole_input': test_diastole, 'metadata_input': test_patient_data}, 
#     test_labels,
#     verbose=1
# )
print("Resultados no conjunto de teste:", results)

y_test_pred = model.predict({'image_input': test_images, 'metadata_input': test_patient_data}, batch_size=6)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)


gc.collect()
