import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from Classification3D.models.models import dualInput_Resnet
from Classification3D.utils import *
from Classification3D.preprocessing.load_mms import load_mms_data_dual_input
from Classification3D.preprocessing.load_data import load_acdc_data_dual_input
from Classification3D.preprocessing.loadIncor import load_incor_dual
from keras.optimizers import Adam

model = dualInput_Resnet()

# Carregue os pesos do modelo treinado
model_weights_path = os.path.join(WEIGHT_PATH, 'multipleD_accuracy.weights.keras')
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"Pesos carregados de: {model_weights_path}")
else:
    print(f"ERRO: Arquivo de pesos não encontrado em {model_weights_path}")
    exit()

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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

resultsMMs = model.evaluate({'systole_input': test_systole_MMs, 'diastole_input': test_diastole_MMs}, 
    labelsMMs, verbose=1)

resultsAcdc = model.evaluate({'systole_input': test_systole_Acdc, 'diastole_input': test_diastole_Acdc}, 
    labelsAcdc, verbose=1)

resultsIncor = model.evaluate({'systole_input': test_systole_Incor, 'diastole_input': test_diastole_Incor}, 
    labelsIncor, verbose=1)

results = model.evaluate({'systole_input': test_systole, 'diastole_input': test_diastole}, 
    test_labels, verbose=1)

print("Fazendo previsões no conjunto de teste...")
y_test_pred_probs = model.predict(
    {'systole_input': test_systole, 'diastole_input': test_diastole},
    batch_size=4
)

y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
y_test_true_classes = np.argmax(test_labels, axis=1)


# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true_classes, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true_classes, y_test_pred_classes, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predMms = model.predict({'systole_input': test_systole_MMs, 'diastole_input': test_diastole_MMs}, batch_size=4)
y_test_pred_classesMms = np.argmax(y_test_predMms, axis=1)
y_test_trueMms = np.argmax(labelsMMs, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueMms, y_test_pred_classesMms)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueMms, y_test_pred_classesMms, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predAcdc = model.predict({'systole_input': test_systole_Acdc, 'diastole_input': test_diastole_Acdc}, batch_size=4)
y_test_pred_classesAcdc = np.argmax(y_test_predAcdc, axis=1)
y_test_trueAcdc = np.argmax(labelsAcdc, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueAcdc, y_test_pred_classesAcdc)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueAcdc, y_test_pred_classesAcdc, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

y_test_predIncor = model.predict({'systole_input': test_systole_Incor, 'diastole_input': test_diastole_Incor}, batch_size=4)
y_test_pred_classesIncor = np.argmax(y_test_predIncor, axis=1)
y_test_trueIncor = np.argmax(labelsIncor, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueIncor, y_test_pred_classesIncor)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueIncor, y_test_pred_classesIncor, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

gc.collect()