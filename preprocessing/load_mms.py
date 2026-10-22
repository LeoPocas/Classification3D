import os
import csv
import cv2
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import *
from Classification3D.preprocessing.augmentation import apply_augmentation
from Classification3D.utils import *

DEFAULT_PREPROCESSING = {
    "apply_roi": False,
    "apply_clahe": False,
    "normalization": None,
    "resampling": False,
    "augmentation": None,
    "augmentation_rate": 1.0,
    "save_debug_images": False,
}
ALLOWED_PATHOLOGIES = {"HCM", "NOR", "DCM"}


def _optional_float(row, key):
    value = row.get(key, "")
    if value is None or not str(value).strip():
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _optional_phase(row, key):
    value = _optional_float(row, key)
    if value is None or not value.is_integer():
        return None
    return int(value)


def _metadata_values(patient_details):
    values = []
    for key in ("weight", "sex", "age"):
        value = patient_details[key]
        values.append(float(value) if value is not None else 0.0)
    return values


def _load_nifti_float32(file_path):
    image = nib.load(file_path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Volume não é 4D (shape={data.shape}): {file_path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Volume contém NaN ou infinito: {file_path}")
    return data, image.header.get_zooms()[:2]


def _valid_phases(end_diastole, end_systole, frame_count):
    return (
        end_diastole is not None
        and end_systole is not None
        and end_diastole >= 0
        and end_systole >= 0
        and end_diastole < frame_count
        and end_systole < frame_count
        and not (end_diastole == 0 and end_systole == 0)
    )


def save_debug_image(volume_3d, patient_id, step_name, output_dir="output/debug_images"):
    """Salva o slice central de um volume 3D para inspeção visual."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        z_center = volume_3d.shape[2] // 2
        if volume_3d.ndim == 4:
            slice_img = volume_3d[:, :, z_center, 0]
        else:
            slice_img = volume_3d[:, :, z_center]

        slice_img = np.nan_to_num(slice_img, nan=0.0, posinf=0.0, neginf=0.0)
        if slice_img.dtype == np.uint8:
            slice_norm = slice_img
        else:
            low, high = np.percentile(slice_img, (1, 99))
            if high > low:
                slice_norm = np.clip((slice_img - low) / (high - low), 0, 1)
                slice_norm = (slice_norm * 255).astype(np.uint8)
            else:
                slice_norm = np.zeros_like(slice_img, dtype=np.uint8)

        filepath = os.path.join(output_dir, f"{patient_id}_{step_name}.png")
        cv2.imwrite(filepath, slice_norm)
    except Exception as error:
        print(f"Erro ao salvar imagem de debug: {error}")


def process_single_volume(volume_3d, target_shape, config, patient_id="unknown", debug_save=False):
    """Aplica ao M&Ms o mesmo pipeline parametrizado usado pelo Incor."""
    aug_config = config.get("augmentation")

    if debug_save:
        save_debug_image(volume_3d, patient_id, "0_raw_aug" if aug_config else "0_raw")

    if aug_config:
        volume_3d = apply_augmentation(volume_3d, aug_config)

    volume_3d = volume_3d.astype(np.float32)
    volume_3d = pad_or_crop_volume(volume_3d, target_shape)

    norm_method = config.get("normalization")
    if norm_method == "min_max":
        v_min, v_max = np.min(volume_3d), np.max(volume_3d)
        if v_max - v_min > 0:
            volume_3d = (volume_3d - v_min) / (v_max - v_min)
        else:
            volume_3d = np.zeros_like(volume_3d)
    elif norm_method == "z_score":
        mean, std = np.mean(volume_3d), np.std(volume_3d)
        if std > 0:
            volume_3d = (volume_3d - mean) / std
        else:
            volume_3d = np.zeros_like(volume_3d)

    if debug_save and norm_method:
        save_debug_image(volume_3d, patient_id, f"1_norm_{norm_method}")

    if config.get("resampling", False):
        pass

    if config.get("apply_clahe", False):
        volume_3d = apply_clahe(volume_3d)

    if debug_save:
        save_debug_image(volume_3d, patient_id, "final_processed")

    if volume_3d.ndim == 3:
        volume_3d = np.repeat(volume_3d[..., np.newaxis], 1, axis=-1)
    return volume_3d

def load_mms_data(
    training=True,
    data_dir=MMs_REESPACADO,
    csv_path=CSV_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    preprocessing_config=None,
):
    if preprocessing_config is None:
        preprocessing_config = DEFAULT_PREPROCESSING.copy()

    volumes = []
    labels = []
    patient_data = []
    debug_saved_count = 0

    # Leitura do arquivo CSV
    patient_info = {}
    excluded_patient_pathologies = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology'].strip().upper()
            if pathology not in ALLOWED_PATHOLOGIES:
                excluded_patient_pathologies[patient_id] = pathology
                continue
            weight = _optional_float(row, "Weight")
            sex = 0 if row['Sex'] == 'M' else 1 if row['Sex'] == 'F' else None  # Converte sexo para numérico
            age = _optional_float(row, "Age")
            label = label_mapping.get(pathology, -1)
            ed = _optional_phase(row, "ED")
            es = _optional_phase(row, "ES")
            patient_info[patient_id] = {
                'label': label,
                'weight': weight,
                'sex': sex,
                'age': age,
                'ed': ed,
                'es': es
            }

    if training:
        print("training:", training)
        folder = 'Training'
    else: 
        folder = 'Testing'
        print("training:", training)
    folder_path = os.path.join(data_dir, folder)

    if os.path.exists(folder_path):
        for patient_id in os.listdir(folder_path):
            patient_path = os.path.join(folder_path, patient_id)

            if patient_id not in patient_info:
                excluded_pathology = excluded_patient_pathologies.get(patient_id)
                if excluded_pathology:
                    print(
                        f"Atenção: Paciente {patient_id} encontrado no CSV, "
                        f"mas ignorado por patologia não selecionada: {excluded_pathology}."
                    )
                else:
                    print(f"Atenção: Dados do paciente {patient_id} não encontrados no CSV.")
                continue

            label = patient_info[patient_id]['label']
            weight = patient_info[patient_id]['weight']
            sex = patient_info[patient_id]['sex']
            age = patient_info[patient_id]['age']
            EndD = patient_info[patient_id]['ed']
            EndS = patient_info[patient_id]['es']
            EndD = int(EndD) if EndD is not None else None
            EndS = int(EndS) if EndS is not None else None
            
            if(label==-1):
                continue
            if not _valid_phases(EndD, EndS, np.inf):
                print(f"Atenção: Paciente {patient_id} ignorado por fases ED/ES inválidas.")
                continue
            
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:  # Ignorar os rótulos, processar só as imagens
                    nii_path = os.path.join(patient_path, filename)
                    try:
                        data_4d, voxel_size = _load_nifti_float32(nii_path)
                    except (OSError, ValueError) as error:
                        print(f"Atenção: arquivo ignorado: {error}")
                        continue
                    if not _valid_phases(EndD, EndS, data_4d.shape[3]):
                        print(f"Atenção: Paciente {patient_id} ignorado por fases ED/ES fora do volume.")
                        continue
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                    if preprocessing_config.get("apply_roi", False):
                        rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                        if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]) or \
                                rect1[0] >= rect2[0] or rect1[1] >= rect2[1]:
                            print("Arquivo ignorado devido a ROI inválida.", filename)
                            continue
                        img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    else:
                        img4D_ROI = data_4d

                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                    # print(data_4d.shape, img4D_ROI.shape, filename)
                    # Extração dos volumes 3D
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == EndD:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                            volume_3d_ED = process_single_volume(
                                volume_3d_ED,
                                target_shape,
                                preprocessing_config,
                                patient_id=f"{patient_id}_ED",
                                debug_save=preprocessing_config.get("save_debug_images", False)
                                and debug_saved_count < 1,
                            )
                            volumes.append(volume_3d_ED)
                            labels.append(label)
                            patient_data.append(_metadata_values(patient_info[patient_id]))
                            
                        elif t == EndS:
                            volume_3d_ES = img4D_ROI[:, :, :, t]
                            volume_3d_ES = process_single_volume(
                                volume_3d_ES,
                                target_shape,
                                preprocessing_config,
                                patient_id=f"{patient_id}_ES",
                                debug_save=preprocessing_config.get("save_debug_images", False)
                                and debug_saved_count < 1,
                            )
                            if preprocessing_config.get("save_debug_images", False):
                                debug_saved_count += 1
                            volumes.append(volume_3d_ES)
                            labels.append(label)
                            patient_data.append(_metadata_values(patient_info[patient_id]))

        # Conversão para arrays numpy
        volumes = np.array(volumes)
        labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
        patient_data = np.asarray(patient_data, dtype=np.float32).reshape(-1, 3)

        return volumes, labels, patient_data


def load_mms_data_dual_input(
    training=True,
    data_path=MMs_PATH,
    csv_path=CSV_PATH,
    target_shape=TARGET_SHAPE,
    label_mapping=LABEL_MAPPING_MMS,
    zoom_factor=ZOOM,
    preprocessing_config=None,
):
    if preprocessing_config is None:
        preprocessing_config = DEFAULT_PREPROCESSING.copy()

    systole_images, diastole_images, labels = [], [], []
    patient_data = []
    debug_saved_count = 0

    patient_info = {}
    excluded_patient_pathologies = {}

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            patient_id = row['External code']
            pathology = row['Pathology'].strip().upper()
            if pathology not in ALLOWED_PATHOLOGIES:
                excluded_patient_pathologies[patient_id] = pathology
                continue
            weight = _optional_float(row, "Weight")
            sex = 0 if row['Sex'] == 'M' else 1 if row['Sex'] == 'F' else None  # Converte sexo para numérico
            age = _optional_float(row, "Age")
            label = label_mapping.get(pathology, -1)
            ed = _optional_phase(row, "ED")
            es = _optional_phase(row, "ES")
            patient_info[patient_id] = {
                'label': label,
                'weight': weight,
                'sex': sex,
                'age': age,
                'ed': ed,
                'es': es
            }

    if training:
        folders = ['Training']
    else: 
        folders = ['Testing']
    for folder in folders:
        path = os.path.join(data_path, folder)
        if not os.path.exists(path):
            print(f"Atenção: A pasta {path} não existe.")
            continue
        patients = os.listdir(path)
        for patient_id in patients:
            patient_path = os.path.join(path, patient_id)

            # Ignorar pacientes não listados no CSV
            if patient_id not in patient_info:
                excluded_pathology = excluded_patient_pathologies.get(patient_id)
                if excluded_pathology:
                    print(
                        f"Atenção: Paciente {patient_id} encontrado no CSV, "
                        f"mas ignorado por patologia não selecionada: {excluded_pathology}."
                    )
                else:
                    print(f"Atenção: Dados do paciente {patient_id} não encontrados no CSV.")
                continue

            patient_details = patient_info[patient_id]
            label = patient_details['label']
            weight = patient_details['weight']
            sex = patient_details['sex']
            age = patient_details['age']
            EndD = patient_details['ed']
            EndS = patient_details['es']
            EndD = int(EndD) if EndD is not None else None
            EndS = int(EndS) if EndS is not None else None

            # Procurar volumes ED (diástole) e ES (sístole)
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz') and 'gt' not in filename:
                    if EndD is None or EndS is None or EndD < 0 or EndS < 0:
                        print(f"Atenção: Paciente {patient_id} ignorado por valores inválidos/ausentes de ED ({EndD}) ou ES ({EndS}).")
                        continue
                    if EndD == 0 and EndS == 0:
                        print(f"Atenção: Paciente {patient_id} ignorado porque ED e ES são ambos 0.")
                        continue
                    file_path = os.path.join(patient_path, filename)
                    try:
                        data_4d, voxel_size = _load_nifti_float32(file_path)
                    except (OSError, ValueError) as error:
                        print(f"Atenção: arquivo ignorado: {error}")
                        continue
                    if not _valid_phases(EndD, EndS, data_4d.shape[3]):
                        print(f"Atenção: Paciente {patient_id} ignorado por fases ED/ES fora do volume.")
                        continue
                    data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                    if preprocessing_config.get("apply_roi", True):
                        rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)
                        if np.array_equal(rect1, [0, 0]) or np.array_equal(rect2, [0, 0]) or \
                                rect1[0] >= rect2[0] or rect1[1] >= rect2[1]:
                            print("Arquivo ignorado devido a ROI inválida.", filename)
                            continue
                        img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                    else:
                        img4D_ROI = data_4d

                    img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])

                    volume_3d_ED = None
                    volume_3d_ES = None
                    for t in range(img4D_ROI.shape[3]):                    
                        if t == EndD:
                            volume_3d_ED = img4D_ROI[:, :, :, t]
                        elif t == EndS:
                            volume_3d_ES = img4D_ROI[:, :, :, t]

                    if volume_3d_ED is not None and volume_3d_ES is not None and label is not None:
                        config_original = preprocessing_config.copy()
                        config_original["augmentation"] = None
                        current_debug = preprocessing_config.get("save_debug_images", False) and debug_saved_count < 1
                        volume_3d_ED = process_single_volume(
                            volume_3d_ED,
                            target_shape,
                            config_original,
                            patient_id=f"{patient_id}_ED_orig",
                            debug_save=current_debug,
                        )
                        volume_3d_ES = process_single_volume(
                            volume_3d_ES,
                            target_shape,
                            config_original,
                            patient_id=f"{patient_id}_ES_orig",
                            debug_save=current_debug,
                        )
                        if current_debug:
                            debug_saved_count += 1

                        diastole_images.append(volume_3d_ED)
                        systole_images.append(volume_3d_ES)
                        labels.append(label)
                        patient_data.append(_metadata_values(patient_details))

                        aug_method = preprocessing_config.get("augmentation")
                        aug_rate = max(0.0, float(preprocessing_config.get("augmentation_rate", 1.0)))
                        guaranteed_augmentations = int(np.floor(aug_rate))
                        if np.random.rand() < aug_rate - guaranteed_augmentations:
                            guaranteed_augmentations += 1

                        if training and aug_method and aug_method != 'None':
                            for _ in range(guaranteed_augmentations):
                                volume_3d_ED_aug = process_single_volume(
                                    img4D_ROI[:, :, :, EndD], target_shape, preprocessing_config
                                )
                                volume_3d_ES_aug = process_single_volume(
                                    img4D_ROI[:, :, :, EndS], target_shape, preprocessing_config
                                )
                                diastole_images.append(volume_3d_ED_aug)
                                systole_images.append(volume_3d_ES_aug)
                                labels.append(label)
                                patient_data.append(_metadata_values(patient_details))

                        volume_3d_ES, volume_3d_ED, label = None, None, None

    # Converte as listas para arrays NumPy
    systole_images = np.asarray(systole_images, dtype=np.float32)
    diastole_images = np.asarray(diastole_images, dtype=np.float32)
    labels = to_categorical(np.array(labels), num_classes=len(label_mapping))
    patient_data = np.asarray(patient_data, dtype=np.float32).reshape(-1, 3)
    print(f"Total de amostras carregadas: {len(labels)}")

    return {'systole': systole_images, 'diastole': diastole_images, 'metadata': patient_data}, labels
