import os
import shutil
from Classification3D.utils import INCOR_RESAMPLED_TESTING_PATH, INCOR_RESAMPLED_TRAINING_PATH
# Números fornecidos
numeros = [
    55, 17, 42, 87, 90, 68, 7, 97, 1, 39, 12, 5, 33, 80, 96, 22, 44, 29, 70, 3, 62, 51, 78, 9, 64, 26, 32, 72, 18, 48,
    192, 119, 205, 246, 180, 267, 128, 157, 233, 198, 283, 104, 176, 140, 271, 222, 212, 153, 229, 182, 120, 106, 258,
    168, 136, 270, 174, 224, 278, 202, 334, 371, 297, 353, 390, 388, 346, 392, 299, 339, 361, 379, 323, 332, 396, 381,
    310, 328, 286, 341, 394, 366, 307, 363, 373, 365, 315, 375, 350, 295
]

# Diretórios
treino = INCOR_RESAMPLED_TRAINING_PATH 
teste = INCOR_RESAMPLED_TESTING_PATH   

# Criar diretório de destino se não existir
os.makedirs(teste, exist_ok=True)
# Mover arquivos
for folder in os.listdir(treino):
    origem = os.path.join(treino, folder)
    destino = os.path.join(teste, folder)
    for numero in numeros:
        for arquivo in os.listdir(origem):
            if 'P'+str(numero)+'.nii' in arquivo or 'PN'+str(numero)+'.nii' in arquivo:  # Verifica se o número está no nome do arquivo
                shutil.move(os.path.join(origem, arquivo), destino)
                print(f"Movido: {arquivo}")