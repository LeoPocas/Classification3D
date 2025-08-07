import numpy as np
import os
import gc
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.optimizers import Adam
from Classification3D.models.models import build_med3d
from Classification3D.utils import (
    LABEL_MAPPING_MMS, WEIGHT_PATH, OUTPUT_PATH, INCOR_RESAMPLED_PATH,
    TARGET_SHAPE, ZOOM, SPACING
)
from Classification3D.preprocessing.loadIncor import load_incor_data_with_filenames

# --- FUNÇÃO DE ANÁLISE AUXILIAR ---
def analyze_group(group_name, y_true, y_pred, filenames, class_names, report_labels):
    """
    Realiza a análise de um subgrupo de dados (ED ou ES).
    Calcula e imprime a matriz de confusão, relatório de classificação,
    acurácia e lista de arquivos classificados incorretamente.
    
    Args:
        group_name (str): Nome do grupo para os títulos (e.g., "ED").
        y_true (np.array): Rótulos verdadeiros do grupo.
        y_pred (np.array): Rótulos previstos para o grupo.
        filenames (list): Lista de nomes de arquivo para o grupo.
        class_names (list): Nomes das classes para o relatório.
        report_labels (list): Índices das classes para o relatório.
    """
    print(f"\n\n{'='*25} ANÁLISE DO GRUPO: {group_name.upper()} {'='*25}")
    
    if len(y_true) == 0:
        print(f"Nenhuma amostra encontrada para o grupo {group_name}. Análise pulada.")
        return

    # Acurácia
    accuracy = accuracy_score(y_true, y_pred)
    correct_predictions = int(accuracy * len(y_true))
    print(f"\nAcurácia do Grupo {group_name}: {accuracy:.4f} ({correct_predictions}/{len(y_true)})")

    # Matriz de Confusão
    print(f"\n--- Matriz de Confusão (Grupo: {group_name}) ---")
    cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    print(cm)

    # Relatório de Classificação
    print(f"\n--- Relatório de Classificação (Grupo: {group_name}) ---")
    cr = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        target_names=class_names,
        zero_division=0
    )
    print(cr)

    # Arquivos Incorretos
    print(f"\n--- Arquivos do Grupo {group_name} Classificados Incorretamente ---")
    misclassified_count = 0
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            misclassified_count += 1
            filename = filenames[i]
            true_label_name = class_names[y_true[i]]
            predicted_label_name = class_names[y_pred[i]]
            print(f"Arquivo: {filename}, Verdadeiro: {true_label_name}, Previsto: {predicted_label_name}")

    if misclassified_count == 0:
        print(f"Nenhum arquivo do grupo {group_name} foi classificado incorretamente. Excelente! 🎉")
    else:
        print(f"\nTotal de arquivos do grupo {group_name} classificados incorretamente: {misclassified_count} de {len(filenames)}")

# --- SCRIPT PRINCIPAL ---

# Verificação de segurança
if not LABEL_MAPPING_MMS:
    print("ERRO: LABEL_MAPPING_MMS não está definido ou está vazio. Verifique Classification3D.utils.")
    exit()

# Ordenar nomes de classe para relatórios consistentes
class_names_sorted = [name for name, index in sorted(LABEL_MAPPING_MMS.items(), key=lambda item: item[1])]
report_labels_indices = list(range(len(class_names_sorted)))

# Carregar dados de teste
print("Carregando dados de teste...")
test_images, test_labels, test_filenames = load_incor_data_with_filenames(
    training=False,
    data_dir=INCOR_RESAMPLED_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    ed_es_file_path=os.path.join(OUTPUT_PATH,'ED_ES_instants.txt')
)
print(f"{len(test_images)} amostras de teste carregadas.")

# Construir e carregar o modelo
model = build_med3d()
print("Modelo build_med3d construído.")

model_weights_path = os.path.join(WEIGHT_PATH, "incor1_0.9.weights.keras")
# model_weights_path = os.path.join(WEIGHT_PATH, "single_input_best_val_loss.weights.keras")
if not os.path.exists(model_weights_path):
    print(f"ERRO: Arquivo de pesos não encontrado em {model_weights_path}")
    exit()

print(f"Carregando pesos do modelo de: {model_weights_path}")
model.load_weights(model_weights_path)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- ANÁLISE GERAL (CONJUNTO DE TESTE COMPLETO) ---
print(f"\n{'#'*20} ANÁLISE GERAL DO CONJUNTO DE TESTE {'#'*20}")
print(f"\nAvaliando o modelo no conjunto de teste completo ({len(test_images)} amostras)...")
results = model.evaluate(test_images, test_labels, verbose=1, batch_size=4)
print(f"Resultados no Conjunto de Teste Completo - Perda: {results[0]:.4f}, Acurácia: {results[1]:.4f}")

print("\nFazendo previsões no conjunto de teste completo...")
y_test_pred_probs = model.predict(test_images, batch_size=4, verbose=1)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
y_test_true_classes = np.argmax(test_labels, axis=1)

# --- SEPARAÇÃO E ANÁLISE POR GRUPO (ED vs ES) ---

print("\nSeparando resultados para os grupos ED e ES...")
ed_indices = [i for i, fname in enumerate(test_filenames) if '_ED' in fname]
es_indices = [i for i, fname in enumerate(test_filenames) if '_ES' in fname]

y_true_ed = y_test_true_classes[ed_indices]
y_pred_ed = y_test_pred_classes[ed_indices]
filenames_ed = [test_filenames[i] for i in ed_indices]

y_true_es = y_test_true_classes[es_indices]
y_pred_es = y_test_pred_classes[es_indices]
filenames_es = [test_filenames[i] for i in es_indices]

analyze_group("ED", y_true_ed, y_pred_ed, filenames_ed, class_names_sorted, report_labels_indices)
analyze_group("ES", y_true_es, y_pred_es, filenames_es, class_names_sorted, report_labels_indices)


# --- ANÁLISE POR PACIENTE (CORRETO EM ED E ES) ---
print(f"\n\n{'='*25} ANÁLISE POR PACIENTE {'='*25}")
patient_analysis = {}

# Agrupa os resultados por ID do paciente
for i, filename in enumerate(test_filenames):
    is_correct = (y_test_pred_classes[i] == y_test_true_classes[i])
    
    patient_id = None
    scan_type = None
    if '_ED' in filename:
        patient_id = filename.split('_ED')[0]
        scan_type = 'ED'
    elif '_ES' in filename:
        patient_id = filename.split('_ES')[0]
        scan_type = 'ES'
    else:
        # Ignora arquivos que não se encaixam no padrão esperado
        continue
    
    if patient_id not in patient_analysis:
        patient_analysis[patient_id] = {'ED': None, 'ES': None}
        
    patient_analysis[patient_id][scan_type] = is_correct

# Calcula as estatísticas
both_correct = 0
ed_only_correct = 0
es_only_correct = 0
both_incorrect = 0
incomplete_patients = 0
complete_patient_ids = []

for patient_id, results in patient_analysis.items():
    ed_status = results.get('ED')
    es_status = results.get('ES')
    
    # Verifica se o paciente tem ambos os scans para uma análise justa
    if ed_status is None or es_status is None:
        incomplete_patients += 1
        continue
    
    complete_patient_ids.append(patient_id)
    if ed_status and es_status:
        both_correct += 1
    elif ed_status and not es_status:
        ed_only_correct += 1
    elif not ed_status and es_status:
        es_only_correct += 1
    else:
        both_incorrect += 1

total_complete_patients = len(complete_patient_ids)
print(f"\nTotal de pacientes únicos com dados completos (ED e ES): {total_complete_patients}")
if incomplete_patients > 0:
    print(f"Aviso: {incomplete_patients} pacientes foram ignorados por não terem ambos os scans ED e ES.")

if total_complete_patients > 0:
    accuracy_patient_level = (both_correct / total_complete_patients) * 100
    print(f"\n- Pacientes com AMBOS os scans (ED e ES) corretos: {both_correct} ({accuracy_patient_level:.2f}%)")
    print(f"- Pacientes com APENAS ED correto: {ed_only_correct}")
    print(f"- Pacientes com APENAS ES correto: {es_only_correct}")
    print(f"- Pacientes com AMBOS os scans incorretos: {both_incorrect}")
else:
    print("\nNenhum paciente com dados completos foi encontrado para a análise.")


# --- LIMPEZA ---
print("\nLimpando memória...")
del test_images, test_labels, test_filenames, model
del y_true_ed, y_pred_ed, filenames_ed, y_true_es, y_pred_es, filenames_es
del patient_analysis
gc.collect()

print("\n--- Fase de Teste Concluída ---")
