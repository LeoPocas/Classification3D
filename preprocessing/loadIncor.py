import os
import re
import csv
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import *
from Classification3D.utils import *

def get_ED_ES_phase_from_file(patient_id, file_path=OUTPUT_PATH+'ED_ES_instants.txt'):
    """
    Retorna os valores de ED_phase e ES_phase de um paciente a partir de um arquivo grande.
    
    :param patient_id: ID do paciente (ex.: "PN23")
    :param file_path: Caminho para o arquivo .txt contendo os dados
    :return: Dicionário com ED_phase e ES_phase, ou uma mensagem de erro
    """
    try:
        # Abrir e ler o arquivo
        with open(file_path, 'r') as file:
            for line in file:
                # Separar os campos da linha
                parts = line.strip().split(",")
                ed_phase = int(re.findall(r'\d+', parts[1])[0])
                es_phase = int(re.findall(r'\d+', parts[2])[0])
                file_name = parts[0].split(': ')[1]

                # ed_phase = int(parts[1].split(': ')[1].strip())  # Converte para int
                # es_phase = int(parts[2].split(': ')[1].strip())  # Converte para int
                
                if file_name == patient_id:
                    return ed_phase, es_phase
        # Caso o ID não seja encontrado
        return {"error": "ID do paciente não encontrado no arquivo"}
    
    except Exception as e:
        return {"error": f"Erro ao ler o arquivo: {str(e)}"}

def load_incor_data(training = True, data_dir=INCOR_RESAMPLED_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
    volumes = []
    labels = []

    if training:
        print("training:", training)
        folder = 'Training'
    else: 
        folder = 'Testing'
        print("training:", training)
    folder_path = os.path.join(data_dir, folder)

    ###1 TAB para o antigo####
    if os.path.exists(folder_path):
        for status in os.listdir(folder_path): 
            status_path = os.path.join(folder_path, status)
            if (status == 'Normal'):
                print(status, status_path)
                label = 0
            elif (status == 'Hipertrófico'):
                print(status, status_path)
                label = 2
            else:
                print(status, status_path)
                label = 1
            for patient_id in os.listdir(status_path):
                patient_path = os.path.join(status_path, patient_id)
                if patient_id.endswith('.nii'):
                    ni_img = nib.load(patient_path)
                    data_4d = ni_img.get_fdata()
                    voxel_size = SPACING[:2]
                    data_4d = np.transpose(data_4d, [3, 2, 1, 0])
                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                    if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]):
                        print("Arquivo ignorado devido a ROI inválida.", patient_id)
                        continue
                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                    ed_phase, es_phase = get_ED_ES_phase_from_file(patient_id.split('.')[0])
                    # print(data_4d.shape, img4D_ROI.shape, patient_id)
                    # Extração dos volumes 3D
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == ed_phase:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                            # volume_3d_ED = (volume_3d_ED - np.min(volume_3d_ED)) / (np.max(volume_3d_ED) - np.min(volume_3d_ED))  
                            volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                            volume_3d_ED = apply_clahe(volume_3d_ED)
                            volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ED)
                            labels.append(label)
                            
                        elif t == es_phase:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            # volume_3d_ES = (volume_3d_ES - np.min(volume_3d_ES)) / (np.max(volume_3d_ES) - np.min(volume_3d_ES))  
                            volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                            volume_3d_ES = apply_clahe(volume_3d_ES)
                            volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                            volumes.append(volume_3d_ES)
                            labels.append(label)
            print(labels.count(label), label)

            # Conversão para arrays numpy
        volumes = np.array(volumes)
        labels = to_categorical(np.array(labels), num_classes=len(label_mapping))

        return volumes, labels

def load_incor_dual(training = True, data_dir=INCOR_RESAMPLED_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING_MMS, zoom_factor=ZOOM):
    systole, diastole = [], []
    labels = []

    if training:
        print("training:", training)
        folder = 'Training'
    else: 
        folder = 'Testing'
        print("training:", training)
    folder_path = os.path.join(data_dir, folder)

    ###1 TAB para o antigo####
    if os.path.exists(folder_path):
        for status in os.listdir(folder_path): 
            status_path = os.path.join(folder_path, status)
            if (status == 'Normal'):
                print(status, status_path)
                label = 0
            elif (status == 'Hipertrófico'):
                print(status, status_path)
                label = 2
            else:
                print(status, status_path)
                label = 1
            for patient_id in os.listdir(status_path):
                patient_path = os.path.join(status_path, patient_id)
                if patient_id.endswith('.nii'):
                    ni_img = nib.load(patient_path)
                    data_4d = ni_img.get_fdata()
                    voxel_size = SPACING[:2]
                    data_4d = np.transpose(data_4d, [3, 2, 1, 0])
                    # Extração das ROIs e cálculo das dimensões
                    rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                    if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]):
                        print("Arquivo ignorado devido a ROI inválida.", patient_id)
                        continue
                    img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                    ed_phase, es_phase = get_ED_ES_phase_from_file(patient_id.split('.')[0])
                    # print(data_4d.shape, img4D_ROI.shape, patient_id)
                    # Extração dos volumes 3D
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == ed_phase:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                            # volume_3d_ED = (volume_3d_ED - np.min(volume_3d_ED)) / (np.max(volume_3d_ED) - np.min(volume_3d_ED))  
                            volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                            volume_3d_ED = apply_clahe(volume_3d_ED)
                            volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                            
                        elif t == es_phase:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            # volume_3d_ES = (volume_3d_ES - np.min(volume_3d_ES)) / (np.max(volume_3d_ES) - np.min(volume_3d_ES))  
                            volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                            volume_3d_ES = apply_clahe(volume_3d_ES)
                            volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                    systole.append(volume_3d_ES)
                    diastole.append(volume_3d_ED)
                    labels.append(label)
            print(labels.count(label), label)

            # Conversão para arrays numpy
        diastole = np.array(diastole)
        systole = np.array(systole)
        labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
 
        return {'systole': systole, 'diastole': diastole}, labels
