import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from Classification3D.models.classification.models import build_med3d, build_med3d_with_ssl
from Classification3D.utils import *
from Classification3D.preprocessing.load_mms import load_mms_data
from Classification3D.preprocessing.load_data import load_3d_roi_sep
from Classification3D.preprocessing.loadIncor import load_incor_data
from keras.optimizers import Adam

model = build_med3d()

model.load_weights(WEIGHT_PATH + 'multiple+incor_resnet.weights.keras')

optimizer = Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

test_imagesMms, test_labelsMms, test_patient_dataMms = load_mms_data(training=False)
test_imagesAcdc, test_labelsAcdc, test_patient_dataAcdc = load_3d_roi_sep(ACDC_REESPACADO_TESTING)
test_imagesIncor, test_labelsIncor = load_incor_data(training=False)

test_images = np.concatenate([test_imagesMms, test_imagesAcdc, test_imagesIncor], axis=0)
test_labels = np.concatenate([test_labelsMms, test_labelsAcdc, test_labelsIncor], axis=0)

results = model.evaluate(test_images, test_labels, verbose=1)
resultsMms = model.evaluate(test_imagesMms, test_labelsMms, verbose=1)
resultsAcdc = model.evaluate(test_imagesAcdc, test_labelsAcdc, verbose=1)
resultsIncor = model.evaluate(test_imagesIncor, test_labelsIncor, verbose=1)

print("Resultados no conjunto de teste:", results)
print("Resultados no subconjunto do MMs de teste:", resultsMms)
print("Resultados no subconjunto do ACDC de teste:", resultsAcdc)
print("Resultados no subconjunto do Incor de teste:", resultsIncor)

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

y_test_predIncor = model.predict(test_imagesIncor, batch_size=6)
y_test_pred_classesIncor = np.argmax(y_test_predIncor, axis=1)
y_test_trueIncor = np.argmax(test_labelsIncor, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_trueIncor, y_test_pred_classesIncor)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_trueIncor, y_test_pred_classesIncor, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)
