import tensorflow as tf
import numpy as np
import os
import gc
from Classification3D.models.models import dualInput_Resnet
from Classification3D.preprocessing.load_mms import load_mms_data_dual_input
from Classification3D.preprocessing.load_data import load_acdc_data_dual_input
from Classification3D.preprocessing.loadIncor import load_incor_dual
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from Classification3D.utils import LABEL_MAPPING_MMS, ACDC_REESPACADO_TESTING, ACDC_REESPACADO_TRAINING, WEIGHT_PATH

# Configuração para usar precisão mista
mixed_precision.set_global_policy('float32')

# class ConfusionMatrixCallback(Callback):
#     def __init__(self, validation_data, batch_size):
#         super().__init__()
#         self.validation_data = validation_data
#         self.batch_size = batch_size

#     def on_epoch_end(self, epoch, logs=None):
#         # Dados de validação: múltiplos inputs e labels
#         x_val_systole, x_val_diastole, y_val = self.validation_data
        
#         # Geração de previsões usando todos os inputs
#         y_pred = self.model.predict(
#             {'systole_input': x_val_systole, 'diastole_input': x_val_diastole},
#             batch_size=self.batch_size
#         )
        
#         # Obter as classes preditas
#         y_pred_classes = np.argmax(y_pred, axis=1)
        
#         # Obter as classes verdadeiras
#         y_true = np.argmax(y_val, axis=1)
        
#         # Gerar a matriz de confusão
#         cm = confusion_matrix(y_true, y_pred_classes)
#         print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
#         # Gerar o relatório de classificação
#         cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING.keys()))
#         print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)
        
dataMMs, labelsMMs = load_mms_data_dual_input()
systole_images_MMs = dataMMs['systole']
diastole_images_MMs = dataMMs['diastole']

dataAcdc, labelsAcdc = load_acdc_data_dual_input(data_path=ACDC_REESPACADO_TRAINING) 
systole_images_acdc = dataAcdc['systole']
diastole_images_acdc = dataAcdc['diastole']

dataIncor, labelsIncor = load_incor_dual() 
systole_images_incor = dataIncor['systole']
diastole_images_incor = dataIncor['diastole']

x_train_systole_MMs, x_val_systole_MMs, x_train_diastole_MMs, x_val_diastole_MMs, y_train_MMs, y_val_MMs= train_test_split(
    systole_images_MMs, diastole_images_MMs, labelsMMs, test_size=0.1, random_state=41
)
x_train_systole_acdc, x_val_systole_acdc, x_train_diastole_acdc, x_val_diastole_acdc, y_train_acdc, y_val_acdc= train_test_split(
    systole_images_acdc, diastole_images_acdc, labelsAcdc, test_size=0.1, random_state=41
)
x_train_systole_incor, x_val_systole_incor, x_train_diastole_incor, x_val_diastole_incor, y_train_incor, y_val_incor= train_test_split(
    systole_images_incor, diastole_images_incor, labelsIncor, test_size=0.1, random_state=41
)

x_train_systole = np.concatenate([x_train_systole_MMs, x_train_systole_acdc, x_train_systole_incor], axis=0)
x_train_diastole = np.concatenate([x_train_diastole_MMs, x_train_diastole_acdc, x_train_diastole_incor], axis=0)
y_train = np.concatenate([y_train_MMs, y_train_acdc, y_train_incor], axis=0)
x_val_systole = np.concatenate([x_val_systole_MMs, x_val_systole_acdc, x_val_systole_incor], axis=0)
x_val_diastole = np.concatenate([x_val_diastole_MMs, x_val_diastole_acdc, x_val_diastole_incor], axis=0)
y_val = np.concatenate([y_val_MMs, y_val_acdc, y_val_incor], axis=0)

print(y_val_MMs.shape, y_val_acdc.shape, y_val_incor.shape)
model = dualInput_Resnet()

# Compilar o modelo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
callbacks = [
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "multipleD_accuracy.weights.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max", verbose=-1
    ),
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "multipleD_loss.weights.keras"),
        save_best_only=True, monitor="val_loss", mode="min", verbose=-1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=4, min_lr=1e-6)
    # ConfusionMatrixCallback(
    #     validation_data=(x_val_systole, x_val_diastole, y_val),  # Adicionar todos os inputs e labels
    #     batch_size=2
    # )
]

history = model.fit(
    {'systole_input': x_train_systole, 'diastole_input': x_train_diastole}, 
    y_train,
    validation_data=(
    {'systole_input': x_val_systole, 'diastole_input': x_val_diastole}, y_val), 
    epochs=300, batch_size=4,
    callbacks=callbacks,
    verbose=2
)

# Testar o modelo com os dados de teste

del x_train_systole, x_train_diastole, y_train, x_val_systole, x_val_diastole, y_val
gc.collect()

dataMMs, labelsMMs = load_mms_data_dual_input(training=False)
test_systole_MMs = dataMMs['systole']
test_diastole_MMs = dataMMs['diastole']

dataAcdc, labelsAcdc = load_acdc_data_dual_input(data_path=ACDC_REESPACADO_TESTING) 
test_systole_Acdc = dataAcdc['systole']
test_diastole_Acdc = dataAcdc['diastole']

dataIncor, labelsIncor = load_incor_dual(training=False) 
test_systole_Incor = dataIncor['systole']
test_diastole_Incor = dataIncor['diastole']

test_systole = np.concatenate([test_systole_MMs, test_systole_Acdc, test_systole_Incor], axis=0)
test_diastole = np.concatenate([test_diastole_MMs, test_diastole_Acdc, test_diastole_Incor], axis=0)
test_labels = np.concatenate([labelsMMs, labelsAcdc, labelsIncor], axis=0)

resultsMMs = model.evaluate(
    {'systole_input': test_systole_MMs, 'diastole_input': test_diastole_MMs}, 
    labelsMMs,
    verbose=1
)

resultsAcdc = model.evaluate(
    {'systole_input': test_systole_Acdc, 'diastole_input': test_diastole_Acdc}, 
    labelsAcdc,
    verbose=1
)

resultsIncor = model.evaluate(
    {'systole_input': test_systole_Incor, 'diastole_input': test_diastole_Incor}, 
    labelsIncor,
    verbose=1
)

results = model.evaluate(
    {'systole_input': test_systole, 'diastole_input': test_diastole}, 
    test_labels,
    verbose=1
)

print("Resultados no conjunto de teste:", results)
print("Resultados no subconjunto do MMs de teste:", resultsMMs)
print("Resultados no subconjunto do ACDC de teste:", resultsAcdc)
print("Resultados no subconjunto do Incor de teste:", resultsIncor)

y_test_pred = model.predict({'systole_input': test_systole, 'diastole_input': test_diastole}, batch_size=4)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predMms = model.predict({'systole_input': test_systole_MMs, 'diastole_input': test_diastole_MMs}, batch_size=4)
y_test_pred_classesMms = np.argmax(y_test_predMms, axis=1)
y_test_trueMms = np.argmax(labelsMMs, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueMms, y_test_pred_classesMms)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueMms, y_test_pred_classesMms, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predAcdc = model.predict({'systole_input': test_systole_Acdc, 'diastole_input': test_diastole_Acdc}, batch_size=4)
y_test_pred_classesAcdc = np.argmax(y_test_predAcdc, axis=1)
y_test_trueAcdc = np.argmax(labelsAcdc, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueAcdc, y_test_pred_classesAcdc)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueAcdc, y_test_pred_classesAcdc, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predIncor = model.predict({'systole_input': test_systole_Incor, 'diastole_input': test_diastole_Incor}, batch_size=4)
y_test_pred_classesIncor = np.argmax(y_test_predIncor, axis=1)
y_test_trueIncor = np.argmax(labelsIncor, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueIncor, y_test_pred_classesIncor)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueIncor, y_test_pred_classesIncor, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

gc.collect()