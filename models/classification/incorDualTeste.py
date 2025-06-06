import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from Classification3D.models.classification.models import dualInput_Resnet
from Classification3D.utils import WEIGHT_PATH, LABEL_MAPPING_MMS
from Classification3D.preprocessing.loadIncor import load_incor_dual_with_filenames
from keras.optimizers import Adam

model = dualInput_Resnet()

# Carregue os pesos do modelo treinado
model_weights_path = os.path.join(WEIGHT_PATH, 'incor2_accuracy.weights.keras') # Ou o nome correto dos seus pesos
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"Pesos carregados de: {model_weights_path}")
else:
    print(f"ERRO: Arquivo de pesos não encontrado em {model_weights_path}")
    exit()

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

test_data_dict, test_labels_categorical, test_filenames = load_incor_dual_with_filenames(training=False)

if len(test_filenames) == 0:
    print("Nenhum dado de teste foi carregado. Verifique o caminho e a estrutura dos dados.")
    exit()

test_systole_volumes = test_data_dict['systole']
test_diastole_volumes = test_data_dict['diastole']

print(f"Número de amostras de teste carregadas: {len(test_filenames)}")

results = model.evaluate(
    {'systole_input': test_systole_volumes, 'diastole_input': test_diastole_volumes},
    test_labels_categorical,
    verbose=1
)
print(f"Resultados da Avaliação no Conjunto de Teste - Perda: {results[0]:.4f}, Acurácia: {results[1]:.4f}")

# --- Previsões e Análise de Erros ---
print("Fazendo previsões no conjunto de teste...")
y_test_pred_probs = model.predict(
    {'systole_input': test_systole_volumes, 'diastole_input': test_diastole_volumes},
    batch_size=4
)

# Converter probabilidades para classes (índices)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
# Converter rótulos one-hot verdadeiros para classes (índices)
y_test_true_classes = np.argmax(test_labels_categorical, axis=1)

# Obter nomes das classes do mapeamento para o relatório
class_names = [name for name, index in sorted(LABEL_MAPPING_MMS.items(), key=lambda item: item[1])]

# Matriz de Confusão
print("\n--- Matriz de Confusão (Conjunto de Teste) ---")
test_cm = confusion_matrix(y_test_true_classes, y_test_pred_classes)
print(test_cm)

# Relatório de Classificação
print("\n--- Relatório de Classificação (Conjunto de Teste) ---")
test_cr = classification_report(y_test_true_classes, y_test_pred_classes, target_names=class_names)
print(test_cr)

# --- Identificar e Listar Arquivos Classificados Incorretamente ---
print("\n--- Arquivos Classificados Incorretamente (Conjunto de Teste) ---")
misclassified_count = 0
for i in range(len(y_test_true_classes)):
    if y_test_pred_classes[i] != y_test_true_classes[i]:
        misclassified_count += 1
        filename = test_filenames[i]
        true_label_name = class_names[y_test_true_classes[i]]
        predicted_label_name = class_names[y_test_pred_classes[i]]
        print(f"Arquivo: {filename}, Rótulo Verdadeiro: {true_label_name}, Rótulo Previsto: {predicted_label_name}")

if misclassified_count == 0:
    print("Nenhum arquivo foi classificado incorretamente no conjunto de teste. Ótimo trabalho!")
else:
    print(f"\nTotal de arquivos classificados incorretamente: {misclassified_count} de {len(test_filenames)}")

del test_systole_volumes, test_diastole_volumes, test_labels_categorical, test_filenames
gc.collect()