import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from Classification3D.models.models import build_med3d
from Classification3D.utils import *
from Classification3D.models.loss import combined_loss
from Classification3D.preprocessing.load_data import load_4d_roi_sep, load_acdc_data_3d, load_3d_roi_sep
from keras.optimizers import Adam

model = build_med3d()

model.load_weights(WEIGHT_PATH + 'single_input_best_val_accuracy.weights.keras')
# model.load_weights(WEIGHT_PATH + 'single_input_best_val_loss.weights.keras')

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# test_images, test_labels = load_4d_roi_sep(ACDC_TESTING_PATH)
# results = model.evaluate(test_images, test_labels, verbose=1)
test_images, test_labels, test_patient_data = load_3d_roi_sep(ACDC_REESPACADO_TESTING)
results = model.evaluate({'input_layer': test_images, 'metadata_input': test_patient_data}, test_labels, verbose=1)

# Fazer as predições no conjunto de teste
predictions = model.predict({'input_layer': test_images, 'metadata_input': test_patient_data}, verbose=1, batch_size=4)

# Transformar as predições para a forma de classes
predictions_classes = np.argmax(predictions, axis=1)

test_labels_classes = np.argmax(test_labels, axis=-1)

# Calcular Accuracy geral
accuracy = accuracy_score(test_labels_classes.flatten(), predictions_classes.flatten())  # Passando os rótulos verdadeiros e previstos
print(f'Accuracy geral: {accuracy:.4f}')

# Matriz de confusão
conf_matrix = confusion_matrix(test_labels_classes.flatten(), predictions_classes.flatten(), labels=range(NUM_CLASSES))  # Passando os rótulos verdadeiros e previstos
print("\nMatriz de Confusão:")
print(conf_matrix)