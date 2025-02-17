import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import zoom
from keras.utils import to_categorical

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
NUM_CLASSES = 5
TARGET_SHAPE = (96, 96, 16)  # Dimensão final padronizada para o modelo

# Variáveis para armazenar as dimensões máximas das máscaras
max_dims = [0, 0, 0]

# ROI padrão para o cenário de teste (baseado em médias do treinamento)
default_roi = ((0, 96), (0, 96), (0, 16))

dataset_path = './ACDC/database/training/'

def calculate_hausdorff_roi(mask):
    coords = np.column_stack(np.where(mask > 0)).astype(float)
    if len(coords) == 0:
        return (0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2])

    # Atualizar as dimensões máximas
    global max_dims
    dims = np.max(coords, axis=0).astype(int) - np.min(coords, axis=0).astype(int)
    max_dims = np.maximum(max_dims, dims)

    # Usar a distância de Hausdorff para otimização
    dists = [directed_hausdorff(coords, np.delete(coords, i, axis=0))[0] for i in range(len(coords))]

    min_coords = np.min(coords, axis=0).astype(int)
    max_coords = np.max(coords, axis=0).astype(int)

    return tuple(min_coords), tuple(max_coords)

def extract_roi(volume, min_coords, max_coords):
    return volume[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]

def resize_volume(volume, target_shape=TARGET_SHAPE):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape[:3])]
    return zoom(volume, zoom_factors, order=1)  # Interpolação linear

def load_acdc_data_with_roi(data_path=dataset_path, label_mapping=LABEL_MAPPING, num_classes=NUM_CLASSES, is_training=True):
    images, labels = [], []
    patients = os.listdir(data_path)

    for patient in patients:
        print(f"Processando paciente: {patient}")
        patient_path = os.path.join(data_path, patient)
        info_file_path = os.path.join(patient_path, 'Info.cfg')

        with open(info_file_path, 'r') as f:
            info = f.readlines()

        label = next((label_mapping.get(line.split(':')[1].strip(), -1) for line in info if 'Group' in line), -1)

        img, mask = None, None
        for filename in os.listdir(patient_path):
            file_path = os.path.join(patient_path, filename)

            if filename.endswith('.nii.gz'):
                if 'gt' in filename:
                    mask = nib.load(file_path).get_fdata()
                elif 'frame' in filename and '4d' not in filename:
                    img = nib.load(file_path).get_fdata()
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if img is not None:
            if is_training and mask is not None:
                min_coords, max_coords = calculate_hausdorff_roi(mask)
                img_roi = extract_roi(img, min_coords, max_coords)
            elif not is_training and mask is not None:
                min_coords, max_coords = calculate_hausdorff_roi(mask)
                img_roi = extract_roi(img, min_coords, max_coords)
            else:
                print("ROI padrão aplicado para validação.")
                img_roi = extract_roi(img, default_roi[0], default_roi[1])

            img_resized = resize_volume(img_roi)
            img_resized = np.repeat(img_resized[..., np.newaxis], 3, axis=-1)  # Ajuste para canais RGB

            images.append(img_resized)
            labels.append(label)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=num_classes)

    print(f"Dimensões máximas das máscaras: Comprimento={max_dims[0]}, Largura={max_dims[1]}, Profundidade={max_dims[2]}")

    return images, labels


images, labels = load_acdc_data_with_roi()
print(f"Imagens: {images.shape}, Labels: {labels.shape}")