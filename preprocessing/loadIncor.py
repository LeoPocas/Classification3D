import os
import re
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

def load_incor_data_with_filenames(
    training=True,
    data_dir=INCOR_RESAMPLED_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    ed_es_file_path=OUTPUT_PATH+'ED_ES_instants.txt'
):
    volumes = []
    labels = []
    filenames = []

    if training:
        print("Modo de carregamento (entrada única): Treinamento")
        folder_name = 'Training'
    else:
        print("Modo de carregamento (entrada única): Teste")
        folder_name = 'Testing'
    
    base_folder_path = os.path.join(data_dir, folder_name)

    if not os.path.exists(base_folder_path):
        print(f"Diretório não encontrado: {base_folder_path}")
        return np.array([]), np.array([]), []

    for status_folder in os.listdir(base_folder_path):
        status_path = os.path.join(base_folder_path, status_folder)
        if not os.path.isdir(status_path):
            continue

        label_val = -1
        if status_folder == 'Normal':
            label_val = label_mapping['NOR']
        elif status_folder == 'Hipertrófico':
            label_val = label_mapping['HCM']
        elif status_folder == 'Dilatados':
            label_val = label_mapping['DCM']
        else:
            print(f"Status desconhecido encontrado: {status_folder}. Ignorando.")
            continue
        
        print(f"Processando status: {status_folder} (Rótulo: {label_val}) em {status_path}")

        for nii_filename in os.listdir(status_path):
            if nii_filename.endswith('.nii') or nii_filename.endswith('.nii.gz'):
                patient_file_path = os.path.join(status_path, nii_filename)
                patient_id_for_phases = nii_filename.split('.nii')[0]

                try:
                    ni_img = nib.load(patient_file_path)
                    data_4d = ni_img.get_fdata()
                except Exception as e:
                    print(f"Erro ao carregar {nii_filename}: {e}. Pulando.")
                    continue
                
                voxel_size = SPACING[:2]
                data_4d = np.transpose(data_4d, [3, 2, 1, 0]) # t, z, y, x

                rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]) or rect1[0] >= rect2[0] or rect1[1] >= rect2[1]:
                    print(f"Arquivo {nii_filename} ignorado devido a ROI inválida: {rect1}, {rect2}")
                    continue
                
                img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0]) # x, y, z, t

                phase_info = get_ED_ES_phase_from_file(patient_id_for_phases, file_path=ed_es_file_path)
                if isinstance(phase_info, dict) and "error" in phase_info:
                    print(f"Erro ao obter fases ED/ES para {nii_filename} (ID: {patient_id_for_phases}): {phase_info['error']}. Pulando.")
                    continue
                ed_phase, es_phase = phase_info
                inter_phase = (ed_phase + es_phase)//2
                if not (0 <= ed_phase < img4D_ROI.shape[3] and 0 <= es_phase < img4D_ROI.shape[3]):
                    print(f"Fases ED ({ed_phase}) ou ES ({es_phase}) fora do intervalo para {nii_filename} (frames: {img4D_ROI.shape[3]}). Pulando.")
                    continue
                
                volume_3d_ED = img4D_ROI[:, :, :, ed_phase]
                volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                volume_3d_ED = apply_clahe(volume_3d_ED)
                volume_3d_ED_processed = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
                volumes.append(volume_3d_ED_processed)
                labels.append(label_val)
                filenames.append(f"{nii_filename}_ED")

                volume_3d_IE = img4D_ROI[:, :, :, inter_phase]
                volume_3d_IE = pad_or_crop_volume(volume_3d_IE, target_shape)
                volume_3d_IE = apply_clahe(volume_3d_IE)
                volume_3d_IE_processed = np.repeat(volume_3d_IE[..., np.newaxis], 1, axis=-1)
                volumes.append(volume_3d_IE_processed)
                labels.append(label_val)
                filenames.append(f"{nii_filename}_IE")
            
                volume_3d_ES = img4D_ROI[:, :, :, es_phase]
                volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                volume_3d_ES = apply_clahe(volume_3d_ES)
                volume_3d_ES_processed = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)
                volumes.append(volume_3d_ES_processed)
                labels.append(label_val)
                filenames.append(f"{nii_filename}_ES")
        
        current_label_count = sum(1 for lbl in labels if lbl == label_val)
        print(f"Total de amostras para status {status_folder} (Rótulo {label_val}): {current_label_count}")

    volumes_np = np.array(volumes)
    labels_np = np.array(labels)
    
    if len(labels_np) > 0:
        labels_categorical = to_categorical(labels_np, num_classes=len(label_mapping))
    else:
        labels_categorical = np.array([]).reshape(0, len(label_mapping))

    print(f"Carregamento (entrada única) concluído. Total de volumes: {len(volumes_np)}")
    print(f"Shape dos volumes: {volumes_np.shape}")
    print(f"Shape dos rótulos: {labels_categorical.shape}")
    print(f"Total de nomes de arquivos coletados: {len(filenames)}")
    
    return volumes_np, labels_categorical, filenames

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

def load_incor_dual_with_filenames(
    training=True,
    data_dir=INCOR_RESAMPLED_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    ed_es_file_path=OUTPUT_PATH+'ED_ES_instants.txt'
):
    systole_volumes, diastole_volumes = [], []
    labels = []
    filenames = [] 

    if training:
        print("Modo de carregamento: Treinamento")
        folder_name = 'Training'
    else:
        print("Modo de carregamento: Teste")
        folder_name = 'Testing'
    
    base_folder_path = os.path.join(data_dir, folder_name)

    if not os.path.exists(base_folder_path):
        print(f"Diretório não encontrado: {base_folder_path}")
        return {'systole': np.array([]), 'diastole': np.array([])}, np.array([]), []

    for status_folder in os.listdir(base_folder_path):
        status_path = os.path.join(base_folder_path, status_folder)
        if not os.path.isdir(status_path):
            continue

        label = -1
        if status_folder == 'Normal':
            label = label_mapping['NOR'] 
        elif status_folder == 'Hipertrófico':
            label = label_mapping['HCM']
        elif status_folder == 'Dilatados':
            label = label_mapping['DCM']
        else:
            print(f"Status desconhecido encontrado: {status_folder}. Ignorando.")
            continue
        
        print(f"Processando status: {status_folder} (Rótulo: {label}) em {status_path}")

        for nii_filename in os.listdir(status_path):
            if nii_filename.endswith('.nii') or nii_filename.endswith('.nii.gz'):
                patient_file_path = os.path.join(status_path, nii_filename)
                patient_id_for_phases = nii_filename.split('.nii')[0]

                try:
                    ni_img = nib.load(patient_file_path)
                    data_4d = ni_img.get_fdata()
                except Exception as e:
                    print(f"Erro ao carregar {nii_filename}: {e}. Pulando.")
                    continue

                voxel_size = SPACING[:2] 
                data_4d = np.transpose(data_4d, [3, 2, 1, 0]) # t, z, y, x

                rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]) or rect1[0] >= rect2[0] or rect1[1] >= rect2[1]:
                    print(f"Arquivo {nii_filename} ignorado devido a ROI inválida: {rect1}, {rect2}")
                    continue
                
                img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0]) # x, y, z, t

                phase_info = get_ED_ES_phase_from_file(patient_id_for_phases, file_path=ed_es_file_path)
                if isinstance(phase_info, dict) and "error" in phase_info:
                    print(f"Erro ao obter fases ED/ES para {nii_filename} (ID: {patient_id_for_phases}): {phase_info['error']}. Pulando.")
                    continue
                ed_phase, es_phase = phase_info
                
                if not (0 <= ed_phase < img4D_ROI.shape[3] and 0 <= es_phase < img4D_ROI.shape[3]):
                    print(f"Fases ED ({ed_phase}) ou ES ({es_phase}) fora do intervalo para {nii_filename} (total de frames: {img4D_ROI.shape[3]}). Pulando.")
                    continue

                volume_3d_ED_processed = None
                volume_3d_ES_processed = None

                # Extração dos volumes 3D
                
                # Frame da Diástole Final (ED)
                volume_3d_ED = img4D_ROI[:, :, :, ed_phase]
                volume_3d_ED = pad_or_crop_volume(volume_3d_ED, target_shape)
                volume_3d_ED = apply_clahe(volume_3d_ED)
                volume_3d_ED_processed = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1) 

                # Frame da Sístole Final (ES)
                volume_3d_ES = img4D_ROI[:, :, :, es_phase]
                volume_3d_ES = pad_or_crop_volume(volume_3d_ES, target_shape)
                volume_3d_ES = apply_clahe(volume_3d_ES)
                volume_3d_ES_processed = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)

                if volume_3d_ED_processed is not None and volume_3d_ES_processed is not None:
                    systole_volumes.append(volume_3d_ES_processed)
                    diastole_volumes.append(volume_3d_ED_processed)
                    labels.append(label)
                    filenames.append(nii_filename) 
                else:
                    print(f"Não foi possível processar ED ou ES para {nii_filename}. Pulando.")
        
        current_label_count = sum(1 for lbl in labels if lbl == label)
        print(f"Total de amostras para status {status_folder} (Rótulo {label}): {current_label_count}")


    # Conversão para arrays numpy
    diastole_volumes_np = np.array(diastole_volumes)
    systole_volumes_np = np.array(systole_volumes)
    labels_np = np.array(labels)
    
    if len(labels_np) > 0:
        labels_categorical = to_categorical(labels_np, num_classes=len(label_mapping))
    else:
        labels_categorical = np.array([]).reshape(0, len(label_mapping))


    print(f"Carregamento concluído. Total de amostras: {len(filenames)}")
    print(f"Shape dos volumes de sístole: {systole_volumes_np.shape}")
    print(f"Shape dos volumes de diástole: {diastole_volumes_np.shape}")
    print(f"Shape dos rótulos: {labels_categorical.shape}")
    
    return {'systole': systole_volumes_np, 'diastole': diastole_volumes_np}, labels_categorical, filenames