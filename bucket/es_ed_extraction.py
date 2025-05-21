import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
from natsort import natsorted
from Classification3D.utils import INCOR_RESAMPLED_PATH

def segment_heart(img4D):
    """Segmenta o ventrículo esquerdo aplicando técnicas de limiarização e fechamento morfológico."""
    segmented = np.zeros_like(img4D)

    for t in range(img4D.shape[0]):  # Percorre os instantes de tempo
        for s in range(img4D.shape[1]):  # Percorre as fatias
            slice_img = img4D[t, s]

            # Aplicar limiarização para realçar regiões de interesse
            threshold = np.percentile(slice_img, 95)
            binarized = slice_img > threshold

            # Fechamento morfológico para suavizar segmentação
            closed = binary_closing(binarized)

            # Preencher possíveis buracos na máscara segmentada
            segmented[t, s] = binary_fill_holes(closed)

    return segmented

def extract_phases_from_segmented(segmented_heart):
    """Extrai os instantes de ED e ES com base no volume segmentado do ventrículo esquerdo."""
    volumes = np.sum(segmented_heart, axis=(1, 2, 3))  # Soma apenas o ventrículo segmentado

    ED_phase = np.argmax(volumes)  # Final da diástole (maior volume)
    ES_phase = np.argmin(volumes)  # Final da sístole (menor volume)
    
    return ED_phase, ES_phase

def process_patients(dataset_path):
    """Percorre todas as subpastas do dataset e extrai ED e ES para arquivos .nii"""
    results = []

    for phase in ["Testing", "Training"]:
        for cardiomyopathy in natsorted(os.listdir(os.path.join(dataset_path, phase))):
            cardiomyopathy_path = os.path.join(dataset_path, phase, cardiomyopathy)
            nii_files = natsorted(glob(os.path.join(cardiomyopathy_path, "*.nii")))

            for nii_path in nii_files:
                img_nii = nib.load(nii_path)
                img4D = img_nii.get_fdata()
                img4D = np.transpose(img4D, (3, 2, 1, 0))
                print(img4D.shape)
                segmented_heart = segment_heart(img4D)
                ed_phase, es_phase = extract_phases_from_segmented(segmented_heart)

                results.append([phase, cardiomyopathy, os.path.basename(nii_path), ed_phase, es_phase])

    # Salvar resultados em CSV
    df = pd.DataFrame(results, columns=["Phase", "Cardiomyopathy", "File", "ED_Phase", "ES_Phase"])
    df.to_csv("ED_ES_phases.csv", index=False)
    print("✅ Arquivo ED_ES_phases.csv criado com sucesso!")

# process_patients(INCOR_RESAMPLED_PATH)

import pandas as pd

# Carregar o CSV gerado anteriormente
df = pd.read_csv("ED_ES_phases.csv")

# Selecionar apenas as colunas relevantes
df_filtered = df[["File", "ED_Phase", "ES_Phase"]]

# Salvar em um arquivo TXT
df_filtered.to_csv("ED_ES_instants.txt", sep=",", index=False, header=False)

print("✅ Arquivo 'ED_ES_instants.txt' criado com sucesso!")