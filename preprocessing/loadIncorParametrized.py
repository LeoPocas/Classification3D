import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import apply_clahe, pad_or_crop_volume
from Classification3D.preprocessing.augmentation import apply_augmentation
from Classification3D.utils import (
    INCOR_RESAMPLED_PATH, TARGET_SHAPE, LABEL_MAPPING_MMS, ZOOM, SPACING, OUTPUT_PATH
)
flag_print_ROI, flag_print_normalization, flag_print_clahe = True, True, True # Flags para controle de prints de debug (ROI, Normalização, CLAHE, Pad/Crop)
# Constantes padrão para defaults
DEFAULT_PREPROCESSING = {
    "apply_roi"         : False,
    "apply_clahe"       : False,
    "normalization"     : None, # 'min_max', 'z_score', or None/False
    "resampling"        : False,
    "augmentation"      : None, # 'rotate', 'zoom', 'rotate+zoom', or None
    "augmentation_rate" : 1.0,  # 1.0 = 100% dos dados originais geram uma cópia aumentada
    "save_debug_images" : False # Nova flag default
}

def get_ED_ES_phase_from_file_v2(patient_id, file_path):
    """
    Retorna os valores de ED_phase e ES_phase de um paciente.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(",")
                file_name = parts[0].split(': ')[1]
                
                if file_name == patient_id:
                    ed_phase = int(re.findall(r'\d+', parts[1])[0])
                    es_phase = int(re.findall(r'\d+', parts[2])[0])
                    return ed_phase, es_phase
        return {"error": "ID do paciente não encontrado no arquivo"}
    except Exception as e:
        return {"error": f"Erro ao ler o arquivo: {str(e)}"}

def save_debug_image(volume_3d, patient_id, step_name, output_dir="output/debug_images"):
    """
    Salva o slice central de um volume 3D para inspeção visual.
    
    Args:
        volume_3d: Volume 3D numpy array
        patient_id: Identificador do paciente
        step_name: Nome do passo (ex: 'after_norm', 'raw')
        output_dir: Diretório de salvamento
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Pega o slice central do eixo Z (profundidade)
        # Assumindo shape (H, W, D) ou (H, W, D, 1)
        z_center = volume_3d.shape[2] // 2
        
        # Remove canal extra se existir
        if volume_3d.ndim == 4:
            slice_img = volume_3d[:, :, z_center, 0]
        else:
            slice_img = volume_3d[:, :, z_center]
            
        # Normaliza para 0-255 para visualização correta
        if slice_img.max() - slice_img.min() > 0:
            slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
            slice_norm = (slice_norm * 255).astype(np.uint8)
        else:
            slice_norm = slice_img.astype(np.uint8)

        # Salva usando matplotlib/cv2
        filename = f"{patient_id}_{step_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, slice_norm)
        # print(f"[DEBUG] Imagem salva: {filepath}")
        
    except Exception as e:
        print(f"Erro ao salvar imagem de debug: {e}")

def process_single_volume(volume_3d, target_shape, config, patient_id="unknown", debug_save=False):
    """
    Aplica a cadeia de pré-processamento configurada em um único volume 3D.
    """
    
    # 0. Data Augmentation (Aplicar antes de normalizar ou depois? 
    # Geralmente em imagens médicas, rotações podem introduzir zeros nas bordas (artefatos de interpolação).
    # Se fizermos antes do crop final e normalização, podemos mitigar isso.
    # No entanto, se o Augmentation for on-the-fly durante treino, ele geralmente vem aqui.
    # Se este loader carrega TUDO para memória antes do treino, então o augmentation aqui
    # será aplicado UMA VEZ e fixado para todo o treino (Static Augmentation).
    # SE O OBJETIVO É AUMENTAR O DATASET (Offline Augmentation), ok.
    # SE O OBJETIVO É DYNAMIC AUGMENTATION (Keras Generator), este não é o lugar ideal, 
    # mas dado que carregar 3D é pesado, talvez 'static' seja o desejado pelo usuário agora.
    
    aug_config = config.get("augmentation", None)
    
    
    # Debug: Salva estado inicial (pós-aug se houver)
    if debug_save:
        save_debug_image(volume_3d, patient_id, "0_raw_aug" if aug_config else "0_raw")

    # 1. Normalização
    norm_method = config.get("normalization", "min_max")

    if aug_config:
        # Nota: Idealmente augmentation é aplicado APÓS carregar, mas ANTES de normalizar/pad 
        # para garantir que os valores nulos de rotação sejam tratados.
        # Mas aqui, `volume_3d` já pode ter vindo de um ROI crop.
        volume_3d = apply_augmentation(volume_3d, aug_config)

    # Garantir float32 antes de normalizar para evitar erros de tipo
    volume_3d = volume_3d.astype(np.float32)
    volume_3d = pad_or_crop_volume(volume_3d, target_shape)

    if norm_method == "min_max":
        global flag_print_normalization
        if flag_print_normalization:
            print(f"Aplicando Normalização Min-Max no volume 3D")
            flag_print_normalization = False
        v_min, v_max = np.min(volume_3d), np.max(volume_3d)
        if v_max - v_min > 0:
            volume_3d = (volume_3d - v_min) / (v_max - v_min)
        else:
            # Fallback para evitar NaNs se volume for constante
            volume_3d = np.zeros_like(volume_3d)
            
    elif norm_method == "z_score":
        mean, std = np.mean(volume_3d), np.std(volume_3d)
        if std > 0:
            volume_3d = (volume_3d - mean) / std
        else:
            volume_3d = np.zeros_like(volume_3d)

    if debug_save and norm_method:
        save_debug_image(volume_3d, patient_id, f"1_norm_{norm_method}")
    
    # 3. Resampling (Placeholder: implementação futura se flag ativa)
    if config.get("resampling", False):
        pass 

    # 4. CLAHE
    if config.get("apply_clahe", True):
        # CLAHE espera uint8 ou converte internamente? 
        # Se normalizamos antes para 0-1 (min_max), apply_clahe precisa lidar com isso.
        # Caso o apply_clahe da sua lib faça cast para uint8 (0-255), ok.
        
        # apply_clahe original converte? Vamos checar equalizacao.py: 
        # "volume_normalized = (volume - volume.min()) /..." 
        # Ele re-normaliza internamente, então é seguro.
        
        global flag_print_clahe
        if flag_print_clahe:
            print(f"Aplicando CLAHE no volume 3D")
            flag_print_clahe = False
        volume_3d = apply_clahe(volume_3d)
        
    if debug_save:
        save_debug_image(volume_3d, patient_id, "final_processed")

    # Adicionar dimensão do canal (H, W, D) -> (H, W, D, 1)
    if volume_3d.ndim == 3:
        volume_3d = np.repeat(volume_3d[..., np.newaxis], 1, axis=-1)
        
    return volume_3d

def load_incor_dual_parametrized(
    training=True,
    data_dir=INCOR_RESAMPLED_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    ed_es_file_path=OUTPUT_PATH+'ED_ES_instants.txt',
    preprocessing_config=None
):
    """
    Loader parametrizável para dados duais (Sístole + Diástole).
    Substitui load_incor_dual e permite ativar/desativar etapas.
    """
    
    if preprocessing_config is None:
        preprocessing_config = DEFAULT_PREPROCESSING.copy()
        
    systole_volumes, diastole_volumes = [], []
    labels = []
    # Embora não retorne, pode ser útil manter lista interna ou logar
    filenames = [] 

    folder_name = 'Training' if training else 'Testing'
    print(f"Modo de carregamento Parametrizado: {folder_name}")
    print(f"Configuração de Preprocessamento: {preprocessing_config}")
    
    base_folder_path = os.path.join(data_dir, folder_name)

    if not os.path.exists(base_folder_path):
        print(f"Diretório não encontrado: {base_folder_path}")
        return {'systole': np.array([]), 'diastole': np.array([])}, np.array([])

    # Flag para controlar salvamento de debug (salva apenas o primeiro paciente do lote)
    should_save_debug = preprocessing_config.get("save_debug_images", False)
    debug_saved_count = 0

    for status_folder in os.listdir(base_folder_path):
        status_path = os.path.join(base_folder_path, status_folder)
        if not os.path.isdir(status_path):
            continue

        # Mapeamento de Labels
        if status_folder == 'Normal':
            label_val = label_mapping.get('NOR', 0)
        elif status_folder == 'Hipertrófico':
            label_val = label_mapping.get('HCM', 2)
        elif status_folder == 'Dilatados':
            label_val = label_mapping.get('DCM', 1)
        else:
            continue # Ignora pastas desconhecidas

        print(f"Processando status: {status_folder}")

        for nii_filename in os.listdir(status_path):
            if nii_filename.endswith('.nii') or nii_filename.endswith('.nii.gz'):
                patient_path = os.path.join(status_path, nii_filename)
                
                try:
                    # Carregamento NIfTI
                    ni_img = nib.load(patient_path)
                    data_4d = ni_img.get_fdata()
                    data_4d = np.transpose(data_4d, [3, 2, 1, 0]) 
                    
                    voxel_size = SPACING[:2]
                    
                    # --- EXTRAÇÃO DE ROI (TOGLEÁVEL) ---
                    if preprocessing_config.get("apply_roi", True):
                        global flag_print_ROI
                        if flag_print_ROI:
                            print(f"Aplicando ROI com voxel size {voxel_size} e zoom {zoom_factor}")
                            flag_print_ROI = False
                        rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                        
                        # Validação ROI
                        if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]) or \
                           rect1[0] >= rect2[0] or rect1[1] >= rect2[1]:
                            # Logar erro silenciosamente ou printar se necessário
                            continue
                        img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    else:
                        img4D_ROI = data_4d # Sem ROI crop

                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0]) 

                    # Obter fases ED/ES
                    patient_id = nii_filename.split('.nii')[0]
                    phase_info = get_ED_ES_phase_from_file_v2(patient_id, ed_es_file_path)
                    
                    if isinstance(phase_info, dict):
                        continue # Erro ao achar fases
                        
                    ed_phase, es_phase = phase_info
                    
                    # Validação de range de frames
                    if not (0 <= ed_phase < img4D_ROI.shape[3] and 0 <= es_phase < img4D_ROI.shape[3]):
                        continue

                    # Extrair volumes crus
                    vol_ED = img4D_ROI[:, :, :, ed_phase]
                    vol_ES = img4D_ROI[:, :, :, es_phase]

                    # Verifica se deve salvar debug para este paciente (apenas o primeiro que der certo)
                    current_debug = False
                    if should_save_debug and debug_saved_count < 1:
                        current_debug = True
                        debug_saved_count += 1
                        print(f"Salvas imagens de debug para o paciente: {patient_id}")

                    # --- PIPELINE DE PROCESSAMENTO (PARAMETRIZADO & AUGMENTATION) ---
                    
                    # 1. Adicionar o dado ORIGINAL (sempre, se possível)
                    # Removemos temporariamente a flag de augmentation para processar o original limpo
                    config_original = preprocessing_config.copy()
                    config_original["augmentation"] = None 
                    
                    vol_ED_orig = process_single_volume(vol_ED, target_shape, config_original, patient_id=f"{patient_id}_ED_orig", debug_save=current_debug)
                    vol_ES_orig = process_single_volume(vol_ES, target_shape, config_original, patient_id=f"{patient_id}_ES_orig", debug_save=current_debug)

                    if vol_ED_orig is not None and vol_ES_orig is not None:
                        systole_volumes.append(vol_ES_orig)
                        diastole_volumes.append(vol_ED_orig)
                        labels.append(label_val)
                        filenames.append(nii_filename)
                        
                        # 2. Adicionar dados AUGMENTADOS (se configurado e for treino)
                        aug_method = preprocessing_config.get("augmentation")
                        aug_rate = preprocessing_config.get("augmentation_rate", 1.0)
                        
                        # Decide aleatoriamente se este paciente sofrerá augmentation baseada na taxa
                        should_augment = np.random.rand() < aug_rate
                        
                        if training and aug_method and aug_method != 'None' and should_augment:
                            # Aplica augmentation N vezes por paciente?
                            # Para mestrado, dobrar o dataset (1x aug) é um bom começo.
                            # Vamos gerar 1 variação para cada original.
                            
                            # Config com augmentation ativo (o próprio preprocessing_config já tem)
                            vol_ED_aug = process_single_volume(vol_ED, target_shape, preprocessing_config, patient_id=f"{patient_id}_ED_aug", debug_save=current_debug)
                            vol_ES_aug = process_single_volume(vol_ES, target_shape, preprocessing_config, patient_id=f"{patient_id}_ES_aug", debug_save=current_debug)

                            if vol_ED_aug is not None and vol_ES_aug is not None:
                                systole_volumes.append(vol_ES_aug)
                                diastole_volumes.append(vol_ED_aug)
                                labels.append(label_val) # Mesmo label
                                filenames.append(f"aug_{nii_filename}")

                except Exception as e:
                    print(f"Erro processando {nii_filename}: {e}")
                    continue

    # Conversão final
    systole_np = np.array(systole_volumes)
    diastole_np = np.array(diastole_volumes)
    labels_np = np.array(labels)
    
    if len(labels_np) > 0:
        labels_cat = to_categorical(labels_np, num_classes=len(label_mapping))
    else:
        labels_cat = np.array([]).reshape(0, len(label_mapping))
        
    print(f"Total carregado: {len(filenames)} amostras.")
    
    # IMPORTANTE: Manter compatibilidade com load_incor_dual original que retornava 
    # {'systole': ..., 'diastole': ...}, labels
    # Mas este novo loader é usado preferencialmente pelo Runner.
    # O arquivo original loadIncor.py retorna tuple: (dict, labels) para load_incor_dual
    # e (dict, labels, filenames) para load_incor_dual_with_filenames
    
    # Vamos retornar estrutura completa
    return {'systole': systole_np, 'diastole': diastole_np}, labels_cat, filenames

