import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from Classification3D.models.models import dualInput_Resnet
from Classification3D.models.heatmap import generate_and_save_gradcam_3d 
from Classification3D.utils import WEIGHT_PATH, LABEL_MAPPING, OUTPUT_PATH
from Classification3D.preprocessing.loadIncor import load_incor_dual_with_filenames
from keras.optimizers import Adam
from itertools import cycle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = dualInput_Resnet()
model.summary()

model_weights_path = os.path.join(WEIGHT_PATH, 'incor2_0.93.weights.keras') 
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"Pesos carregados de: {model_weights_path}")
else:
    print(f"ERRO: Arquivo de pesos não encontrado em {model_weights_path}")
    exit()

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'auc'])

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
    batch_size=2,
    verbose=1
)
print(f"Resultados da Avaliação no Conjunto de Teste - Perda: {results[0]:.4f}, Acurácia: {results[1]:.4f}, AUC: {results[2]:.4f}")

y_test_pred_probs = model.predict(
    {'systole_input': test_systole_volumes, 'diastole_input': test_diastole_volumes},
    batch_size=2
)

# Converter probabilidades para classes (índices)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
# Converter rótulos one-hot verdadeiros para classes (índices)
y_test_true_classes = np.argmax(test_labels_categorical, axis=1)

# Obter nomes das classes do mapeamento para o relatório
class_names = [name for name, index in sorted(LABEL_MAPPING.items(), key=lambda item: item[1])]

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

plt.figure(figsize=(10, 8))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 28}, cbar=False)
plt.title('Confusion Matrix', fontsize=32)
plt.ylabel('True label', fontsize=28)
plt.xlabel('Predicted label', fontsize=28)
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()

plt.savefig(OUTPUT_PATH + 'matriz_confusao_absoluta.png', dpi=1000)
plt.show()

plt.figure(figsize=(12, 10))

# Calcular a curva ROC e a área ROC para cada classe
n_classes=len(class_names)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    # test_labels_categorical é o y_true no formato one-hot
    # y_test_pred_probs é o y_score com as probabilidades
    fpr[i], tpr[i], _ = roc_curve(test_labels_categorical[:, i], y_test_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar todas as curvas ROC
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Curva ROC da classe {0} (área = {1:0.3f})'
             ''.format(class_names[i], roc_auc[i]))

# Plotar a linha de palpite aleatório
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Palpite Aleatório')

# Configurações finais do gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falsos Positivos', fontsize=18)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=18)
plt.title('Curva ROC para Múltiplas Classes', fontsize=22)
plt.legend(loc="lower right", fontsize=14)
plt.grid(True)
plt.tight_layout()

# Salvar a figura em um arquivo PNG
output_filename = os.path.join(OUTPUT_PATH, 'curva_roc_multiclasse.png')
plt.savefig(output_filename, dpi=300)
print(f"\nGráfico da Curva ROC salvo em: {output_filename}")

plt.show()

sample_idx_to_visualize=50

systole_sample = np.expand_dims(test_systole_volumes[sample_idx_to_visualize], axis=0).astype('float32')
diastole_sample = np.expand_dims(test_diastole_volumes[sample_idx_to_visualize], axis=0).astype('float32')  
pred_label_idx = y_test_pred_classes[sample_idx_to_visualize]
filename_base = os.path.splitext(os.path.basename(test_filenames[sample_idx_to_visualize]))[0]

gradcam_output_filename = os.path.join(OUTPUT_PATH, f'gradcam_{filename_base}.png')

generate_and_save_gradcam_3d(
    model=model,
    systole_volume=systole_sample,
    diastole_volume=diastole_sample,
    pred_class_idx=pred_label_idx,
    class_names=class_names,
    output_filename=gradcam_output_filename
)