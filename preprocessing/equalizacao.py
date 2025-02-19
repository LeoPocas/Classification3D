import cv2
import numpy as np
from Classification3D.utils import CLIP, GRID

def apply_clahe(volume, clip_limit=CLIP, tile_grid_size=GRID):
    # Aplica CLAHE em cada slice do volume
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_slices = []
    for i in range(volume.shape[2]):
        slice_2d = volume[:, :, i].astype(np.uint8)
        enhanced_slice = clahe.apply(slice_2d)
        enhanced_slices.append(enhanced_slice)
    return np.stack(enhanced_slices, axis=-1)


def pad_or_crop_volume(volume, target_shape):
    # Calcular o padding necessário para cada dimensão
    pad_width = [(max((t - s) // 2, 0), max((t - s + 1) // 2, 0)) for s, t in zip(volume.shape, target_shape)]
    # Aplicar padding
    volume_padded = np.pad(volume, pad_width, mode='constant', constant_values=0)
    # Atualizar a forma após o padding
    padded_shape = volume_padded.shape
    # Recalcular o slicing para o cropping centralizado
    crop_slices = [slice(max((ps - t) // 2, 0), max((ps - t) // 2, 0) + t) 
                   for ps, t in zip(padded_shape, target_shape)]
    # Aplicar cropping
    volume_cropped = volume_padded[crop_slices[0], crop_slices[1], crop_slices[2]]

    return volume_cropped


def pad_or_crop_volume_4d(volume, target_shape):
    current_shape = volume.shape
    padding = []
    cropping = []

    for t, c in zip(target_shape, current_shape):
        if t > c:
            pad_before = (t - c) // 2
            pad_after = t - c - pad_before
            padding.append((pad_before, pad_after))
            cropping.append((0, 0))
        else:
            crop_before = (c - t) // 2
            crop_after = c - t - crop_before
            padding.append((0, 0))
            cropping.append((crop_before, crop_after))

    volume_padded = np.pad(volume, padding, mode='constant', constant_values=0)

    slices = [slice(crop[0], -crop[1] if crop[1] > 0 else None) for crop in cropping]
    volume_cropped = volume_padded[slices[0], slices[1], slices[2]]

    return volume_cropped[:target_shape[0], :target_shape[1], :target_shape[2]]