import os
import shutil
from Classification3D.utils import MMs_PATH, OUTPUT_PATH

# Diretórios
OLD_TEST_DIR = MMs_PATH + 'Testing/'  # Diretório original de testes
RESAMPLED_DIR = OUTPUT_PATH + 'mms_resampled/'  # Diretório reamostrado com todos os arquivos misturados
NEW_TEST_DIR = os.path.join(RESAMPLED_DIR, 'Testing')  # Diretório para as pastas de teste
NEW_TRAIN_DIR = os.path.join(RESAMPLED_DIR, 'Training')  # Diretório para as pastas de treinamento

# Garante que as pastas de saída existam
os.makedirs(NEW_TEST_DIR, exist_ok=True)
os.makedirs(NEW_TRAIN_DIR, exist_ok=True)

# 1. Criar uma lista com os nomes das pastas do conjunto de testes original
test_folders = []
for folder in os.listdir(OLD_TEST_DIR):
    folder_path = os.path.join(OLD_TEST_DIR, folder)
    if os.path.isdir(folder_path):  # Certifica-se de que é uma pasta
        test_folders.append(folder)

print(f"Número de pastas de teste identificadas: {len(test_folders)}")

# 2. Classificar as pastas no diretório reamostrado
for folder in os.listdir(RESAMPLED_DIR):
    source_folder = os.path.join(RESAMPLED_DIR, folder)
    if not os.path.isdir(source_folder):  # Ignora arquivos que não são pastas
        continue

    if folder in test_folders:
        # Caso a pasta esteja na lista de teste, mova para o diretório Testing
        target_folder = os.path.join(NEW_TEST_DIR, folder)
        shutil.move(source_folder, target_folder)
        print(f"Pasta de teste movida: {folder}")
    elif "Training" not in folder and "Testing" not in folder:
        # Caso contrário, mova para o diretório Training
        target_folder = os.path.join(NEW_TRAIN_DIR, folder)
        shutil.move(source_folder, target_folder)
        print(f"Pasta de treinamento movida: {folder}")

print(f"Reorganização concluída!")
print(f"- Pastas de teste em: {NEW_TEST_DIR}")
print(f"- Pastas de treinamento em: {NEW_TRAIN_DIR}")