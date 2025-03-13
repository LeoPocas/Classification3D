import os
import numpy as np
import nibabel as nib
import csv
from preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.utils import *

def save_roi_dimensions(roi_dimensions, patient_data, labels, filenames, output_path):
    with open(output_path, 'w') as f:
        for dimensions, data, label, filename in zip(roi_dimensions, patient_data, labels, filenames):
            pathology, height, weight, sex, age = data  # Dados do paciente
            f.write(f"Dimensions: {dimensions} | Label: {label} | Pathology: {pathology} | Height: {height} | Weight: {weight} | Sex: {sex} | Age: {age} | Filename: {filename}\n")


def load_mmms_and_extract_3d_volumes(data_dir=MMs_TRAINING, csv_path=CSV_PATH, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM, output_path=OUTPUT_PATH + 'roi_dimensions_mms.txt'):
    volumes = []
    labels = []
    roi_dimensions = []
    filenames = []
    patient_data = []
    # Leitura do arquivo CSV para obter os rótulos e dados dos pacientes
    patient_info = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology']
            height = float(row['Height']) if row['Height'] else None
            weight = float(row['Weight']) if row['Weight'] else None
            sex = row['Sex']
            age = float(row['Age']) if row['Age'] else None
            label = label_mapping.get(pathology, -1)  # Mapeia a patologia para classe numérica
            patient_info[patient_id] = {
                'label': label,
                'pathology': pathology,
                'height': height,
                'weight': weight,
                'sex': sex,
                'age': age
            }

    # Processa dados das pastas 'labeled' e 'unlabeled'
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

            patient_details = patient_info[patient_id]
            label = patient_details['label']
            pathology = patient_details['pathology']
            height = patient_details['height']
            weight = patient_details['weight']
            sex = patient_details['sex']
            age = patient_details['age']

            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:  # Ignorar arquivos de rótulo
                    nii_path = os.path.join(patient_path, filename)
                    ni_img = nib.load(nii_path)
                    data_4d = ni_img.get_fdata().astype(np.uint16)
                    voxel_size = ni_img.header.get_zooms()[:2]
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])

                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]

                    roi_dimensions.append(img4D_ROI.shape)
                    filenames.append(filename)
                    # Adicione os volumes 3D extraídos para a lista
                    patient_data.append([pathology, height, weight, sex, age])
                    volumes.append(data_4d)
                    labels.append(label)

    # Salve as dimensões das ROIs
    save_roi_dimensions(roi_dimensions, patient_data,  labels, filenames, output_path)

    return volumes, labels

# Exemplo de uso
load_mmms_and_extract_3d_volumes()
