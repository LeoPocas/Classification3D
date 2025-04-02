import os
import numpy as np
import nibabel as nib
import csv
from preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.utils import *

def save_roi_dimensions(roi_dimensions, patient_data, labels, filenames, output_path):
    with open(output_path, 'w') as f:
        for dimensions, data, label, filename in zip(roi_dimensions, patient_data, labels, filenames):
            weight, sex, age = data  # Dados do paciente
            f.write(f"Dimensions: {dimensions} | Label: {label} | Weight: {weight} | Sex: {sex} | Age: {age} | Filename: {filename}\n")


def load_mmms_and_extract_3d_volumes(data_dir=MMs_PATH, csv_path=CSV_PATH, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM, output_path=OUTPUT_PATH + 'dimensions_mms.txt'):
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

    if True:
        folders = ['Training/Labeled', 'Training/Unlabeled', 'Testing']
    else: 
        folders = ['Validation']
    for folder in folders:
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

def load_mms_data(training = True, data_dir=MMs_REESPACADO, csv_path=CSV_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
    volumes = []
    labels = []
    roi_dimensions = []
    filenames = []
    patient_data = []

    # Leitura do arquivo CSV
    patient_info = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology']
            weight = float(row['Weight']) if row['Weight'] else None
            sex = 0 if row['Sex'] == 'M' else 1 if row['Sex'] == 'F' else None  # Converte sexo para numérico
            age = float(row['Age']) if row['Age'] else None
            label = label_mapping.get(pathology, -1)
            ed = float(row['ED']) if row['ED'] else None
            es = float(row['ES']) if row['ES'] else None
            patient_info[patient_id] = {
                'label': label,
                'weight': weight,
                'sex': sex,
                'age': age,
                'ed': ed,
                'es': es
            }

    # if training:
    #     print("training:", training)
    #     folders = ['Training/Labeled', 'Training/Unlabeled', 'Validation']
    # else: 
    #     folders = ['Testing']
    #     print("training:", training)
    # print(folders)
    # for folder in folders:
    #     folder_path = os.path.join(data_dir, folder)
    #     if not os.path.exists(folder_path):
    #         print(f"Atenção: A pasta {folder} não existe em {data_dir}.")
    #         continue  

    if training:
        print("training:", training)
        folder = 'Training'
    else: 
        folder = 'Testing'
        print("training:", training)
    folder_path = os.path.join(data_dir, folder)

    ###1 TAB para o antigo####
    if os.path.exists(folder_path):
        for patient_id in os.listdir(folder_path):
            patient_path = os.path.join(folder_path, patient_id)

            if patient_id not in patient_info:
                print(f"Atenção: Dados do paciente {patient_id} não encontrados no CSV.")
                continue

            label = patient_info[patient_id]['label']
            weight = patient_info[patient_id]['weight']
            sex = patient_info[patient_id]['sex']
            age = patient_info[patient_id]['age']
            EndD = patient_info[patient_id]['ed']
            EndS = patient_info[patient_id]['es']
            
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:  # Ignorar os rótulos, processar só as imagens
                    nii_path = os.path.join(patient_path, filename)
                    ni_img = nib.load(nii_path)
                    data_4d = ni_img.get_fdata()
                    voxel_size = ni_img.header.get_zooms()[:2]
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                    if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]):
                        print("Arquivo ignorado devido a ROI inválida.", filename)
                        continue
                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                    # print(data_4d.shape, img4D_ROI.shape, filename)
                    # Extração dos volumes 3D
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == EndD:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                            volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ED)
                            labels.append(label)
                            weight = float(weight) if weight is not None else 0.0
                            sex = float(sex) if sex is not None else 0.0
                            age = float(age) if age is not None else 0.0
                            patient_data.append([weight, sex, age])
                            roi_dimensions.append(img4D_ROI.shape)
                            filenames.append(filename)
                            
                        elif t == EndS:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ES)
                            labels.append(label)
                            weight = float(weight) if weight is not None else 0.0
                            sex = float(sex) if sex is not None else 0.0
                            age = float(age) if age is not None else 0.0
                            patient_data.append([weight, sex, age])
                            roi_dimensions.append(img4D_ROI.shape)
                            filenames.append(filename)

        save_roi_dimensions(roi_dimensions, patient_data,  labels, filenames, OUTPUT_PATH + 'dimensions_mms.txt')

        return labels, patient_data

# Exemplo de uso
load_mms_data()
