import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from Classification3D.utils import ACDC_TESTING_PATH, ACDC_TRAINING_PATH, OUTPUT_PATH
# Caminhos de entrada e saída
OUTPUT_DIR = OUTPUT_PATH+'/acdc_resampled/'  # Diretório para salvar os novos arquivos
folders = [ACDC_TESTING_PATH, ACDC_TRAINING_PATH]
# Garante que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_and_save_4d(path_nii, output_path):
    """Reamostra um arquivo 4D NIfTI para 1mm de espaçamento e salva o resultado."""
    ni_img = nib.load(path_nii)
    data = ni_img.get_fdata()
    affine = ni_img.affine
    header = ni_img.header

    # Espaçamento original e novo
    current_spacing = header.get_zooms()[:3]  # Considera apenas os eixos x, y, z
    new_spacing = (1.0, 1.0, current_spacing[2])  # 1mm isotrópico nos eixos x e y
    print(f"Reamostrando {path_nii}: Espaçamento atual {current_spacing} -> Novo {new_spacing}")

    # Fatores de escala
    scale_factors = [
        current_spacing[0] / new_spacing[0], 
        current_spacing[1] / new_spacing[1], 
        current_spacing[2] / new_spacing[2],
        1.0  # Dimensão temporal (não muda)
    ]

    # Reamostragem
    resampled_data = zoom(data, scale_factors, order=3)

    # Ajuste da matriz affine para refletir a reamostragem
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / np.array(scale_factors[:3]))

    # Cria um novo objeto NIfTI com os dados reamostrados
    resampled_img = nib.Nifti1Image(resampled_data, new_affine, header)

    # Salva o arquivo reamostrado no diretório de saída
    nib.save(resampled_img, output_path)
    print(f"Arquivo reamostrado salvo em: {output_path}")

# Processa todos os arquivos 4D no diretório
for folder in folders:
    for patient_folder in os.listdir(folder):
        patient_path = os.path.join(folder, patient_folder)
        if not os.path.isdir(patient_path):
            continue  # Ignora se não for uma pasta

        for filename in os.listdir(patient_path):
            if filename.endswith('.nii.gz') and '4d' in filename and 'gt' not in filename:  # Procura apenas arquivos 4D
                input_file = os.path.join(patient_path, filename)
                output_patient_folder = os.path.join(OUTPUT_DIR, patient_folder)
                os.makedirs(output_patient_folder, exist_ok=True)  # Cria a pasta de saída do paciente

                output_file = os.path.join(output_patient_folder, filename)
                resample_and_save_4d(input_file, output_file)