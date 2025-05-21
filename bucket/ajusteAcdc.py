import os
import shutil
from Classification3D.utils import *

ACDC_TRAINING_PATH_NEW = os.path.join(ACDC_REESPACADO, 'training/')
ACDC_TESTING_PATH_NEW = os.path.join(ACDC_REESPACADO, 'testing/')

# Criar pastas de treinamento e teste dentro de acdc_resampled, se não existirem
os.makedirs(ACDC_TRAINING_PATH_NEW, exist_ok=True)
os.makedirs(ACDC_TESTING_PATH_NEW, exist_ok=True)

# Listar os pacientes na pasta acdc_resampled
patients = sorted(os.listdir(ACDC_REESPACADO))

# Redistribuir pacientes e copiar arquivos Info.cfg
for patient in patients:
    # Ignorar as pastas recém-criadas de treinamento e teste
    if patient in ['training', 'testing']:
        continue
    
    patient_id = int(patient.replace("patient", ""))
    
    # Determinar nova pasta de destino
    if 1 <= patient_id <= 100:
        destination_path = os.path.join(ACDC_TRAINING_PATH_NEW)
    elif 101 <= patient_id <= 150:
        destination_path = os.path.join(ACDC_TESTING_PATH_NEW)
    else:
        print(f"ID do paciente fora do intervalo esperado: {patient_id}")
        continue
    
    # Mover a pasta do paciente para o destino
    shutil.move(os.path.join(ACDC_REESPACADO, patient), destination_path)
    
    # Buscar e copiar o arquivo Info.cfg dos caminhos antigos
    old_patient_path = os.path.join(
        ACDC_TRAINING_PATH if patient in os.listdir(ACDC_TRAINING_PATH) else ACDC_TESTING_PATH, patient
    )
    info_file_path = os.path.join(old_patient_path, 'Info.cfg')
    
    if os.path.exists(info_file_path):
        shutil.copy(info_file_path, os.path.join(destination_path, patient))
    else:
        print(f"Info.cfg não encontrado para {patient}")

print("Pacientes redistribuídos e arquivos Info.cfg copiados com sucesso!")