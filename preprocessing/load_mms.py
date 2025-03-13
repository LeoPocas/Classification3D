import os
import csv
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import *
from Classification3D.utils import *

def load_mmms_data(data_dir=MMs_TRAINING, csv_path=CSV_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
    volumes = []
    labels = []
    patient_data = []

    # Leitura do arquivo CSV
    patient_info = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology']
            weight = float(row['Weight']) if row['Weight'] else None
            sex = row['Sex']
            age = float(row['Age']) if row['Age'] else None
            label = label_mapping.get(pathology, -1)  # Mapeia a patologia para classe numérica
            patient_info[patient_id] = {
                'label': label,
                'pathology': pathology,
                'weight': weight,
                'sex': sex,
                'age': age
            }

    for folder in ['Labeled', 'Unlabeled']:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Atenção: A pasta {folder} não existe em {data_dir}.")
            continue

        for patient_id in os.listdir(folder_path):
            patient_path = os.path.join(folder_path, patient_id)

            if patient_id not in patient_info:
                print(f"Atenção: Dados do paciente {patient_id} não encontrados no CSV.")
                continue

            label = patient_info[patient_id]['label']
            weight = patient_info[patient_id]['weight']
            sex = patient_info[patient_id]['sex']
            age = patient_info[patient_id]['age']

            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:  # Ignorar os rótulos, processar só as imagens
                    nii_path = os.path.join(patient_path, filename)
                    ni_img = nib.load(nii_path)
                    data_4d = ni_img.get_fdata().astype(np.uint16)
                    voxel_size = ni_img.header.get_zooms()[:2]
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                    
                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])

                    # Extração dos volumes 3D
                    for t in range(img4D_ROI.shape[3]):
                        volume_3d = img4D_ROI[:, :, :, t]
                        # Aplicação do CLAHE
                        volume_3d = apply_clahe(volume_3d)
                        volume_3d = pad_or_crop_volume(volume_3d, target_shape)
                        volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))      
                        volume_3d = volume_3d[..., np.newaxis]
                        
                        volumes.append(volume_3d)
                        labels.append(label)
                        patient_data.append([weight, sex, age])

    # Conversão para arrays numpy
    volumes = np.array(volumes)
    labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
    patient_data = np.array(patient_data)

    return volumes, labels, patient_data
