import tensorflow as tf
import numpy as np
import gc
import os
from Classification3D.models.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet
from Classification3D.preprocessing.load_data import load_4d_and_extract_3d_volumes, load_4d_roi_sep, load_3d_roi_sep, load_acdc_data_3d, load_acdc_data_dual_input
from Classification3D.preprocessing.load_mms import load_mms_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from Classification3D.models.loss import combined_loss, focal_loss
from sklearn.preprocessing import StandardScaler
from ...utils import LABEL_MAPPING, WEIGHT_PATH, ACDC_REESPACADO_TESTING

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
            {'input_layer': x_val_img, 'metadata_input': x_val_meta},
            batch_size=self.batch_size
        )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING.keys()))
        print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)

# Carregar os dados
images, labels, patient_data = load_3d_roi_sep()
# images, labels, patient_data = load_acdc_data_3d()
# data, labels = load_acdc_data_dual_input(data_path=ACDC_TRAINING_PATH) 
# systole_images = data['systole']
# diastole_images = data['diastole']
# patient_data = data['metadata']
# print(patient_data)
# Normalização dos metadados (peso e altura)]
# scaler = StandardScaler()
# patient_data = scaler.fit_transform(patient_data)

# x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=35)
x_train_img, x_val_img, y_train, y_val, x_train_meta, x_val_meta = train_test_split(
    images, labels, patient_data, test_size=0.3, random_state=45
)
# x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val, x_train_meta, x_val_meta = train_test_split(
#     systole_images, diastole_images, labels, patient_data, test_size=0.2, random_state=42
# )

# model = cnn_3d_model()
model = build_med3d()
# model = newModel()
# model = dualInput_Resnet()

# Compilar o modelo
optimizer = Adam(learning_rate=0.00006)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "acdc_best_val_accuracy.weights.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max", verbose=-1
    ),
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "acdc_best_val_loss.weights.keras"),
        save_best_only=True, monitor="val_loss", mode="min", verbose=-1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=4, min_lr=1e-6)
    # ConfusionMatrixCallback(validation_data=(x_val_img, x_val_meta, y_val), batch_size=6)
]
# callbacks = [
#     ModelCheckpoint(WEIGHT_PATH + "med3d_dual_input.weights.keras", save_best_only=True, monitor="val_loss"),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
#     # ConfusionMatrixCallback(
#     #     validation_data=(x_val_systole, x_val_diastole, x_val_meta, y_val),  # Adicionar todos os inputs e labels
#     #     batch_size=20
#     # )
# ]


# Treinar o modelo
history = model.fit(
    x_train_img,
    y_train,
    validation_data=(x_val_img, y_val),
    epochs=500, batch_size=16,
    callbacks=callbacks,
    verbose=2
)

# Treinar o modelo
# history = model.fit(
#     {'systole_input': x_train_systole, 'diastole_input': x_train_diastole, 'metadata_input': x_train_meta}, 
#     y_train,
#     validation_data=(
#         {'systole_input': x_val_systole, 'diastole_input': x_val_diastole, 'metadata_input': x_val_meta}, y_val), 
#     epochs=50, batch_size=8,
#     callbacks=callbacks,
#     verbose=2
# )

# Testar o modelo com os dados ajustados
# test_images, test_labels, test_patient_data = load_acdc_data_3d(ACDC_TESTING_PATH)
test_images, test_labels, test_patient_data = load_3d_roi_sep(ACDC_REESPACADO_TESTING)
# test_data, test_labels = load_acdc_data_dual_input(data_path=ACDC_TESTING_PATH) 
# test_systole = test_data['systole']
# test_diastole = test_data['diastole']
# test_patient_data = test_data['metadata']


results = model.evaluate({'input_layer': test_images, 'metadata_input': test_patient_data}, test_labels, verbose=1)
# results = model.evaluate(
#     {'systole_input': test_systole, 'diastole_input': test_diastole, 'metadata_input': test_patient_data}, 
#     test_labels,
#     verbose=1
# )
print(results)

y_test_pred = model.predict({'input_layer': test_images, 'metadata_input': test_patient_data}, batch_size=6)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)


gc.collect()
