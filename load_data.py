import nibabel as nib
import numpy as np
import os
from keras.utils import to_categorical
from roiExtraction import get_ROI_distance_transform

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
TARGET_SHAPE = (96, 96, 12)
NUM_CLASSES = 5
MAX_TIME_DIM = 16
dataset_path = './ACDC/database/training/'

def load_4d_and_extract_3d_volumes(data_dir=dataset_path, label_mapping=LABEL_MAPPING, apply_padding_cropping=True, target_shape=TARGET_SHAPE):
    volumes = []
    labels = []

    for patient_id in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_id)

        info_file_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_file_path, 'r') as f:
            info = f.readlines()
        label = None
        for line in info:
            if 'Group' in line:
                label = line.split(':')[1].strip()
                break
        label = label_mapping.get(label, -1)

        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz') and '4d' in filename and 'gt' not in filename:
                nii_path = os.path.join(patient_path, filename)
                ni_img = nib.load(nii_path)
                data_4d = ni_img.get_fdata().astype(np.uint16)

                # Extração dos volumes 3D
                for t in range(data_4d.shape[3]):
                    volume_3d = data_4d[:, :, :, t]
                    if apply_padding_cropping:
                        volume_3d = pad_or_crop_volume(volume_3d, target_shape)

                    volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))      
                    volume_3d = np.repeat(volume_3d[..., np.newaxis], 3, axis=-1)

                    volumes.append(volume_3d)
                    labels.append(label) 

    images = np.array(volumes)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return images, labels

def load_acdc_data_3d(data_path=dataset_path, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING, num_classes=NUM_CLASSES):
    images, labels = [], []
    patients = os.listdir(data_path)

    for patient in patients:
        patient_path = os.path.join(data_path, patient)
        info_file_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_file_path, 'r') as f:
            info = f.readlines()
        label = None
        for line in info:
            if 'Group' in line:
                label = line.split(':')[1].strip()
                break
        label = label_mapping.get(label, -1)

        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz') and '4d' not in filename and 'gt' not in filename:
                file_path = os.path.join(patient_path, filename)
                img = nib.load(file_path).get_fdata()
                img = (img - np.min(img)) / (np.max(img) - np.min(img))            
                img_adjusted = pad_or_crop_volume(img, target_shape)
                img_adjusted = np.repeat(img_adjusted[..., np.newaxis], 3, axis=-1)
                images.append(img_adjusted)
                labels.append(label)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return images, labels

def load_acdc_data_4d(data_path=dataset_path, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING, num_classes=NUM_CLASSES):
    images, labels = [], []
    patients = os.listdir(data_path)

    for patient in patients:
        patient_path = os.path.join(data_path, patient)
        info_file_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_file_path, 'r') as f:
            info = f.readlines()
        label = None
        for line in info:
            if 'Group' in line:
                label = line.split(':')[1].strip()
                break
        label = label_mapping.get(label, -1)

        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz') and '4d' in filename and 'gt' not in filename:
                file_path = os.path.join(patient_path, filename)
                img_4d = nib.load(file_path).get_fdata()
                img_4d = (img_4d - np.min(img_4d)) / (np.max(img_4d) - np.min(img_4d))  # Normalização por exame

                # Ajuste da dimensão temporal
                total_frames = img_4d.shape[-1]
                if total_frames > MAX_TIME_DIM:
                    start_frame = (total_frames - MAX_TIME_DIM) // 2
                    end_frame = start_frame + MAX_TIME_DIM
                    img_4d = img_4d[..., start_frame:end_frame]

                adjusted_frames = [pad_or_crop_volume_4d(img_4d[..., t], target_shape) for t in range(img_4d.shape[-1])]
                adjusted_frames = np.stack(adjusted_frames, axis=0)
                adjusted_frames = np.repeat(adjusted_frames[..., np.newaxis], 3, axis=-1)

                # Padding para a dimensão do tempo
                if adjusted_frames.shape[0] < MAX_TIME_DIM:
                    padding = ((0, MAX_TIME_DIM - adjusted_frames.shape[0]), (0, 0), (0, 0), (0, 0), (0, 0))
                    adjusted_frames = np.pad(adjusted_frames, padding, mode='constant', constant_values=0)
                else:
                    adjusted_frames = adjusted_frames[:MAX_TIME_DIM]

                images.append(adjusted_frames)
                labels.append(to_categorical(label, num_classes=num_classes))

    return np.array(images), np.array(labels)

def load_4d_roi_sep(data_dir=dataset_path, target_shape=TARGET_SHAPE, label_mapping = LABEL_MAPPING, voxel_size=None, zoom_factor=1.2):
    volumes = []
    labels = []

    for patient_id in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_id)

        info_file_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_file_path, 'r') as f:
            info = f.readlines()
        label = None
        for line in info:
            if 'Group' in line:
                label = line.split(':')[1].strip()
                break
        label = label_mapping.get(label, -1)

        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz') and '4d' in filename and 'gt' not in filename:
                nii_path = os.path.join(patient_path, filename)
                ni_img = nib.load(nii_path)
                data_4d = ni_img.get_fdata().astype(np.uint16)
                voxel_size = ni_img.header.get_zooms()[0:2]
                data_4d = np.transpose(data_4d, [3, 2, 0, 1])
                # Extração das ROIs e cálculo das dimensões
                rect1, rect2 = get_ROI_distance_transform(data_4d, voxel_size, zoom_factor)

                img4D_ROI = data_4d[:, :, rect1[0]:rect2[0], rect1[1]:rect2[1]]
                img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
                # Extração dos volumes 3D
                for t in range(img4D_ROI.shape[3]):
                    volume_3d = img4D_ROI[:, :, :, t]
                    volume_3d = pad_or_crop_volume(volume_3d, target_shape)

                    volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))      
                    volume_3d = np.repeat(volume_3d[..., np.newaxis], 3, axis=-1)

                    volumes.append(volume_3d)
                    labels.append(label)
                
    volumes =  np.array(volumes)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return volumes, labels

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