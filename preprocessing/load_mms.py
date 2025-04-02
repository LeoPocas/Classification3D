import os
import csv
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import *
from Classification3D.utils import *

def load_mms_data(training = True, data_dir=MMs_REESPACADO, csv_path=CSV_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
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
            
            if(label==-1):
                continue
            
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
                            # volume_3d_ED = (volume_3d_ED - np.min(volume_3d_ED)) / (np.max(volume_3d_ED) - np.min(volume_3d_ED))  
                            volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                            volume_3d_ED = apply_clahe(volume_3d_ED)
                            volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ED)
                            labels.append(label)
                            weight = float(weight) if weight is not None else 0.0
                            sex = float(sex) if sex is not None else 0.0
                            age = float(age) if age is not None else 0.0
                            patient_data.append([weight, sex, age])
                            
                        elif t == EndS:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            # volume_3d_ES = (volume_3d_ES - np.min(volume_3d_ES)) / (np.max(volume_3d_ES) - np.min(volume_3d_ES))  
                            volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                            volume_3d_ES = apply_clahe(volume_3d_ES)
                            volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ES)
                            labels.append(label)
                            weight = float(weight) if weight is not None else 0.0
                            sex = float(sex) if sex is not None else 0.0
                            age = float(age) if age is not None else 0.0
                            patient_data.append([weight, sex, age])

        # Conversão para arrays numpy
        volumes = np.array(volumes)
        labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
        patient_data = np.array(patient_data)

        return volumes, labels, patient_data


def load_mms_data_dual_input(training = True, data_path=MMs_PATH, csv_path=CSV_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
    systole_images, diastole_images, labels = [], [], []
    patient_data = []

    patient_info = {}

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology']
            weight = float(row['Weight']) if row['Weight'] else None
            sex = 0 if row['Sex'] == 'M' else 1 if row['Sex'] == 'F' else None  # Converte sexo para numérico
            age = float(row['Age']) if row['Age'] else None
            label = label_mapping.get(pathology, 0)
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

    if training:
        folders = ['Training/Labeled', 'Training/Unlabeled', 'Testing']
    else: 
        folders = ['Validation']
    for folder in folders:
        path = os.path.join(data_path, folder)
        patients = os.listdir(path)
        for patient_id in patients:
            patient_path = os.path.join(path, patient_id)

            # Ignorar pacientes não listados no CSV
            if patient_id not in patient_info:
                print(f"Atenção: Dados do paciente {patient_id} não encontrados no CSV.")
                continue

            patient_details = patient_info[patient_id]
            label = patient_details['label']
            weight = patient_details['weight']
            sex = patient_details['sex']
            age = patient_details['age']
            EndD = patient_details['ed']
            EndS = patient_details['es']

            # Procurar volumes ED (diástole) e ES (sístole)
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:
                    if EndD is None or EndS is None or EndD == EndS or EndD < 0 or EndS < 0:
                        print(f"Atenção: Paciente {patient_id} ignorado devido a valores ausentes de ED: {EndD} ou ES: {EndS}.")
                        continue
                    file_path = os.path.join(patient_path, filename)
                    ni_img = nib.load(file_path)
                    data_4d = ni_img.get_fdata().astype(np.uint16)
                    voxel_size = ni_img.header.get_zooms()[0:2]
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)

                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                    # Extração dos volumes 3D
                    # print(str(EndD) + " " + str(EndS) + ' ' + str(range(img4D_ROI.shape[0])) + ' ' + file_path)
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == EndD:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                            # volume_3d_ED = apply_clahe(volume_3d_ED)
                            volume_3d_ED = (volume_3d_ED - np.min(volume_3d_ED)) / (np.max(volume_3d_ED) - np.min(volume_3d_ED))  
                            volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                            volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                            
                        elif t == EndS:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            # volume_3d_ES = apply_clahe(volume_3d_ES)
                            volume_3d_ES = (volume_3d_ES - np.min(volume_3d_ES)) / (np.max(volume_3d_ES) - np.min(volume_3d_ES))  
                            volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                            volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                    if (volume_3d_ED is not None and volume_3d_ES is not None and label is not None):
                        diastole_images.append(volume_3d_ED)
                        systole_images.append(volume_3d_ES)
                        labels.append(label)
                        patient_data.append([weight, sex, age])
                        volume_3d_ES, volume_3d_ED, label = None, None, None

    # Converte as listas para arrays NumPy
    systole_images = np.array(systole_images)
    diastole_images = np.array(diastole_images)
    labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
    patient_data = np.array(patient_data)

    return {'systole': systole_images, 'diastole': diastole_images, 'metadata': patient_data}, labels
