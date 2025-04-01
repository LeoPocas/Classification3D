import os
import numpy as np
import nibabel as nib
from Classification3D.preprocessing.equalizacao import apply_clahe, pad_or_crop_volume
from Classification3D.utils import DATASETS_PATH, TARGET_SHAPE

def load_nifti(file_path):
    nii_img = nib.load(file_path)
    volume_4d = nii_img.get_fdata()
    return volume_4d

def extract_3d_frames(volume_4d, strategy):
    """
    Converte um volume 4D para 3D.
    Estratégias:
    - 'all': usa todos os frames.
    - 'systole_diastole': tenta selecionar final de sístole e diástole.
    """
    if strategy == 'all':
        return [volume_4d[..., i] for i in range(volume_4d.shape[-1])]
    elif strategy == 'systole_diastole':
        return [volume_4d[..., 0], volume_4d[..., -1]]  # Exemplo com primeiro e último frame
    elif isinstance(strategy, int) and strategy > 0:
        return [volume_4d[..., i] for i in range(0, volume_4d.shape[-1], strategy)]
    else:
        raise ValueError("Estratégia desconhecida")

def normalize(volume):
    """Normaliza o volume para a faixa [0, 1]."""
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

def load_kaggle_data(input_folder=DATASETS_PATH+'kaggled4D', strategy=3, clahe=True, target_shape=TARGET_SHAPE):
    processed_volumes = []
    
    for file in os.listdir(input_folder):
        if file.endswith(".nii"):
            file_path = os.path.join(input_folder, file)
            volume_4d = load_nifti(file_path)
            frames_3d = extract_3d_frames(volume_4d, strategy)
            
            for frame in frames_3d:
                normalized = normalize(frame)
                normalized = pad_or_crop_volume(normalized, target_shape)
                if clahe:
                    normalized = apply_clahe(normalized)
                normalized = np.repeat(normalized[..., np.newaxis], 1, axis=-1)    
                processed_volumes.append(normalized)
    
    return np.array(processed_volumes)
