import numpy as np
import os
import gc
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam
from Classification3D.models.models import build_med3d
from Classification3D.utils import (
    LABEL_MAPPING_MMS, WEIGHT_PATH, OUTPUT_PATH, INCOR_RESAMPLED_PATH,
    TARGET_SHAPE, ZOOM, SPACING
)
from Classification3D.preprocessing.loadIncor import load_incor_data_with_filenames

if not LABEL_MAPPING_MMS:
    print("ERRO: LABEL_MAPPING_MMS não está definido ou está vazio. Verifique Classification3D.utils.")
    exit()

class_names_sorted = [name for name, index in sorted(LABEL_MAPPING_MMS.items(), key=lambda item: item[1])]
report_labels_indices = list(range(len(class_names_sorted)))

test_images, test_labels, test_filenames = load_incor_data_with_filenames(
    training=False,
    data_dir=INCOR_RESAMPLED_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    ed_es_file_path=os.path.join(OUTPUT_PATH,'ED_ES_instants.txt')
)

model = build_med3d()
print("Modelo build_med3d construído.")

# model_weights_path = os.path.join(WEIGHT_PATH, "single_input_best_val_accuracy.weights.keras")
model_weights_path = os.path.join(WEIGHT_PATH, "single_input_best_val_loss.weights.keras")

if not os.path.exists(model_weights_path):
    print(f"ERRO: Arquivo de pesos não encontrado em {model_weights_path}")
    print("Certifique-se de que o caminho e o nome do arquivo de pesos estão corretos.")
    exit()

print(f"Carregando pesos do modelo de: {model_weights_path}")
model.load_weights(model_weights_path)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nAvaliando o modelo no conjunto de teste ({len(test_images)} amostras)...")
results = model.evaluate(test_images, test_labels, verbose=1, batch_size=4)
print(f"Resultados no Conjunto de Teste - Perda: {results[0]:.4f}, Acurácia: {results[1]:.4f}")

print("\nFazendo previsões no conjunto de teste...")
y_test_pred_probs = model.predict(test_images, batch_size=4)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
y_test_true_classes = np.argmax(test_labels, axis=1)

print("\n--- Matriz de Confusão (Conjunto de Teste) ---")
test_cm = confusion_matrix(y_test_true_classes, y_test_pred_classes, labels=report_labels_indices)
print(test_cm)

print("\n--- Relatório de Classificação (Conjunto de Teste) ---")
test_cr = classification_report(
    y_test_true_classes,
    y_test_pred_classes,
    labels=report_labels_indices,
    target_names=class_names_sorted,
    zero_division=0
)
print(test_cr)

print("\n--- Arquivos de Teste Classificados Incorretamente ---")
misclassified_test_count = 0
if test_filenames is not None and len(test_filenames) > 0:
    for i in range(len(y_test_true_classes)):
        if y_test_pred_classes[i] != y_test_true_classes[i]:
            misclassified_test_count += 1
            filename = test_filenames[i] # Nomes já incluem _ED ou _ES da função de carregamento
            true_label_name = class_names_sorted[y_test_true_classes[i]] if y_test_true_classes[i] < len(class_names_sorted) else f"Índice_{y_test_true_classes[i]}"
            predicted_label_name = class_names_sorted[y_test_pred_classes[i]] if y_test_pred_classes[i] < len(class_names_sorted) else f"Índice_{y_test_pred_classes[i]}"
            print(f"Arquivo: {filename}, Verdadeiro: {true_label_name}, Previsto: {predicted_label_name}")

    if misclassified_test_count == 0:
        print("Nenhum arquivo de teste classificado incorretamente. Excelente! 🎉")
    else:
        print(f"\nTotal de arquivos de teste classificados incorretamente: {misclassified_test_count} de {len(test_filenames)}")
else:
    print("Nenhum nome de arquivo de teste disponível para análise de erros.")

del test_images, test_labels, test_filenames, model
gc.collect()

print("\n--- Fase de Teste Concluída ---")