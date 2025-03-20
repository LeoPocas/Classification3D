import os
import numpy as np
import pydicom
import nibabel as nib
import shutil
from scipy.spatial.distance import euclidean
from Classification3D.utils import KAGGLE_PATH, OUTPUT_PATH

# Lista de slices a serem removidas (badList)
badList = [("sax_10", "484"), ("sax_7", "282"), ("sax_8", "282"), ("sax_9", "282"), ("sax_10", "11"),
            ("sax_15", "436"), ("sax_8", "436"), ("sax_7", "436"), ("sax_36", "436"), ("sax_21", "282"),
            ("sax_23", "241"), ("sax_90", "195"), ("sax_77", "195"), ("sax_92", "195"), ("sax_80", "195"),
            ("sax_20", "232"), ("sax_8", "393"), ("sax_7", "416"), ("sax_37", "466"), ("sax_16", "280"),
            ("sax_17", "280"), ("sax_18", "280"), ("sax_20", "442"), ("sax_21", "442"), ("sax_22", "442"),
            ("sax_23", "442"), ("sax_24", "442"), ("sax_65", "274"), ("sax_66", "274"), ("sax_67", "274"),
            ("sax_5", "409"), ("sax_6", "409"), ("sax_35", "41"), ("sax_3", "41")]

def load_dicom_series(folder):
    slices = []
    positions = []
    times = []
    for file in sorted(os.listdir(folder), key=lambda x: int(x.split('-')[-1].split('.')[0])):
        if file.endswith(".dcm"):
            dicom_path = os.path.join(folder, file)
            dicom_data = pydicom.dcmread(dicom_path)
            if hasattr(dicom_data, "ImagePositionPatient"):
                slices.append(dicom_data.pixel_array)
                positions.append(np.array(dicom_data.ImagePositionPatient, dtype=np.float32))
                times.append(dicom_data.InstanceNumber)
    if slices:
        return np.stack(slices), positions, times
    return None, None, None

def process_patient(patient_id, patient_folder, output_dir):
    sax_folders = [f for f in os.listdir(patient_folder) if f.startswith("sax_")]
    sax_data = []

    for sax in sax_folders:
        if (sax, patient_id) in badList:
            continue  # Removendo slices da badList
        
        sax_path = os.path.join(patient_folder, sax)
        images, positions, times = load_dicom_series(sax_path)
        
        if images is not None and len(images) > 0:
            # Calcula a média da posição z para esta pasta
            mean_position_z = np.mean([pos[2] for pos in positions])
            sax_data.append((positions, images, sax, mean_position_z))
    
    if not sax_data:
        return  # Nenhuma slice válida para esse paciente
    
    # Ordena as pastas pelo eixo Z (posição espacial)
    sax_data.sort(key=lambda x: x[3])  # Ordena com base na média do eixo z
    valid_slices = []
    valid_positions = []
    mean_distance = np.mean([euclidean(sax_data[i][0][0], sax_data[i-1][0][0]) for i in range(1, len(sax_data))])
    
    for i, (positions, images, sax, mean_position_z) in enumerate(sax_data):
        if i == 0 or not valid_positions:
            reference = sax_data[i+1][0][0] if i + 1 < len(sax_data) else None
        else:
            reference = valid_positions[-1]
        
        if reference is not None:
            distance = euclidean(positions[0], reference)
            if mean_distance * 0.6 <= distance <= mean_distance * 1.4 and mean_distance < 20:
                valid_positions.append(positions[0])
                valid_slices.append(images)
    
    if valid_slices:
        try: 
            # Empilha as fatias válidas em um volume 4D
            valid_slices = np.stack(valid_slices, axis=-1)  # Empilha ao longo do eixo 0
            valid_slices = np.transpose(valid_slices, [2, 1, 3, 0])  # Transposição: [altura, largura, profundidade]
            
            # Matriz affine (ajustando a profundidade)
            affine = np.eye(4)
            scale_factor = 256 / valid_slices.shape[2]  # Ajuste proporcional para profundidade
            affine[2, 2] = scale_factor  # Define o "tamanho" do voxel no eixo z (profundidade)

            # Salva o volume NIfTI
            nib.save(nib.Nifti1Image(valid_slices, affine), os.path.join(output_dir, f"{patient_id}_4d.nii"))
            print(f"Volume 4D salvo para paciente {valid_slices.shape} {patient_id} em {os.path.join(output_dir, f'{patient_id}_4d.nii')}")
        except: 
            print(f"{patient_id} com formato incoerente de dados")


# Diretórios de entrada e saída
input_dir = KAGGLE_PATH+'train/train'
output_dir = OUTPUT_PATH+'kaggled4D'
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove toda a pasta de saída
os.makedirs(output_dir, exist_ok=True)

for patient in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, patient)
    patient_path =  os.path.join(folder_path, 'study')
    if os.path.isdir(patient_path):
        process_patient(patient, patient_path, output_dir)