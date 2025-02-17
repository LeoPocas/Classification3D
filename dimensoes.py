import os
import numpy as np
import nibabel as nib
from roiExtraction import get_ROI_distance_transform

dataset_path = './ACDC/database/training/'
LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

def save_roi_dimensions(roi_dimensions, filenames, output_path):
    with open(output_path, 'w') as f:
        for dimensions, filename in zip(roi_dimensions, filenames):
            f.write(f"{dimensions} patient {filename}\n")

def load_4d_and_extract_3d_volumes(data_dir, label_mapping, apply_padding_cropping=True, target_shape=None, voxel_size=None, zoom_factor=1.2, output_path='roi_dimensions.txt'):
    volumes = []
    labels = []
    roi_dimensions = []
    filenames = []

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

                roi_dimensions.append(img4D_ROI.shape)
                filenames.append(filename)
                # Adicione a volumetria 3D para a lista de volumes
                volumes.append(data_4d)
                labels.append(label)

    # Salve as dimensões das ROIs
    save_roi_dimensions(roi_dimensions, filenames, output_path)

    return volumes, labels

load_4d_and_extract_3d_volumes(dataset_path, LABEL_MAPPING)
