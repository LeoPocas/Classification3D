import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from Classification3D.models.classification.models import build_med3d, build_med3d_with_ssl
from Classification3D.utils import *
from Classification3D.models.loss import combined_loss
from Classification3D.preprocessing.load_mms import load_mms_data, load_mms_data_pure
from keras.optimizers import Adam

model = build_med3d()
# model = build_med3d_with_ssl()

model.load_weights(WEIGHT_PATH + 'mms_resnet.weights.67acc.soVolume.keras')

optimizer = Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# test_images, test_labels = load_4d_roi_sep(ACDC_TESTING_PATH)
# results = model.evaluate(test_images, test_labels, verbose=1)
test_images, test_labels, test_patient_data = load_mms_data_pure(training=False)

print(type(test_images), test_images.dtype)
print(type(test_patient_data), test_patient_data.dtype)
print(type(test_labels), test_labels.dtype)

results = model.evaluate({'image_input': test_images, 'metadata_input': test_patient_data}, test_labels, verbose=1)

# Fazer as predições no conjunto de teste
predictions = model.predict({'image_input': test_images, 'metadata_input': test_patient_data}, verbose=1, batch_size=4)

# Transformar as predições para a forma de classes
predictions_classes = np.argmax(predictions, axis=1)

test_labels_classes = np.argmax(test_labels, axis=-1)

# Calcular Accuracy geral
accuracy = accuracy_score(test_labels_classes.flatten(), predictions_classes.flatten())  # Passando os rótulos verdadeiros e previstos
print(f'Accuracy geral: {accuracy:.4f}')

# Matriz de confusão
conf_matrix = confusion_matrix(test_labels_classes.flatten(), predictions_classes.flatten(), labels=range(len(LABEL_MAPPING_MMS)))  # Passando os rótulos verdadeiros e previstos
print("\nMatriz de Confusão:")
print(conf_matrix)