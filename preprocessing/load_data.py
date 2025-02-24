import nibabel as nib
import numpy as np
import os
import re
from keras.utils import to_categorical
from preprocessing.roiExtraction import get_ROI_distance_transform
from preprocessing.equalizacao import *
from Classification3D.utils import *

def load_4d_and_extract_3d_volumes(data_dir=ACDC_TRAINING_PATH, label_mapping=LABEL_MAPPING, apply_padding_cropping=True, target_shape=TARGET_SHAPE):
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

def load_acdc_data_3d(data_path=ACDC_TRAINING_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING, num_classes=NUM_CLASSES):
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

def load_acdc_data_4d(data_path=ACDC_TRAINING_PATH, target_shape=TARGET_SHAPE, label_mapping=LABEL_MAPPING, num_classes=NUM_CLASSES):
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

def load_4d_roi_sep(data_dir=ACDC_TRAINING_PATH, target_shape=TARGET_SHAPE, label_mapping = LABEL_MAPPING, voxel_size=None, zoom_factor=ZOOM):
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
                    # Aplicação do CLAHE
                    volume_3d = apply_clahe(volume_3d)
                    volume_3d = pad_or_crop_volume(volume_3d, target_shape)
                    volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))      
                    volume_3d = volume_3d[..., np.newaxis]
                    
                    volumes.append(volume_3d)
                    labels.append(label)
                
    volumes =  np.array(volumes)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return volumes, labels

def load_3d_roi_sep(data_dir=ACDC_TRAINING_PATH, target_shape=TARGET_SHAPE, label_mapping = LABEL_MAPPING, voxel_size=None, zoom_factor=ZOOM):
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

        target_frames = get_target_frames(patient_path)

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
                    if (t + 1) in target_frames:  # +1 pois t começa em 0 e os frames começam em 1
                        volume_3d = img4D_ROI[:, :, :, t]
                        # Aplica CLAHE (opcional, conforme discutido anteriormente)
                        volume_3d = apply_clahe(volume_3d)
                        volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))  
                        volume_3d = pad_or_crop_volume(volume_3d, target_shape)
                        volume_3d = np.repeat(volume_3d[..., np.newaxis], 1, axis=-1)
                        volumes.append(volume_3d)
                        labels.append(label)
                
    volumes =  np.array(volumes)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return volumes, labels

def get_target_frames(patient_path):
    target_frames = []
    for filename in os.listdir(patient_path):
        # Procura por arquivos que terminam com '_gt.nii'
        if filename.endswith('_gt.nii.gz'):
            # Extrai o número do frame usando regex
            match = re.search(r'_frame(\d+)_gt', filename)
            if match:
                frame_number = int(match.group(1))
                target_frames.append(frame_number)
    return target_frames