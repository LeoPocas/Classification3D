import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from Classification3D.models.classification.models import build_med3d, build_med3d_with_ssl, dualInput_Resnet
from Classification3D.utils import *
from Classification3D.models.loss import combined_loss
from Classification3D.preprocessing.loadIncor import load_incor_data, load_incor_dual
from keras.optimizers import Adam

# model = build_med3d()
model = dualInput_Resnet()

# model.load_weights(WEIGHT_PATH + 'incor_resnet.weights.keras')
model.load_weights(WEIGHT_PATH + 'incor_dual_input.weights.keras')
optimizer = Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# test_images, test_labels = load_incor_data(training=False)

# results = model.evaluate(test_images, test_labels, verbose=1)

# print("Resultados no conjunto de teste:", results)

# y_test_pred = model.predict(test_images, batch_size=6)
# y_test_pred_classes = np.argmax(y_test_pred, axis=1)
# y_test_true = np.argmax(test_labels, axis=1)

# # Gerar matriz de confusão
# test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
# print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# # Relatório de classificação
# test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
# print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

test_data, test_labels = load_incor_dual(training=False) 
test_systole = test_data['systole']
test_diastole = test_data['diastole']

# results_train = model.evaluate(x_val_img, y_val, verbose=0)
results = model.evaluate(
    {'systole_input': test_systole, 'diastole_input': test_diastole}, 
    test_labels,
    verbose=1
)

# print("Resultados do melhor modelo no conjunto de treino:", results_train)

# del x_train_img, x_val_img, y_train, y_val
# gc.collect()

# results = model.evaluate(test_images, test_labels, verbose=1)

print("Resultados no conjunto de teste:", results)

y_test_pred = model.predict({'systole_input': test_systole, 'diastole_input': test_diastole}, batch_size=6)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING_MMS.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)
