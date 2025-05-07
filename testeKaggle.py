import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
import nibabel as nib
from utils import OUTPUT_PATH
path_nii = './output/kaggled4D/128_4d.nii'
matplotlib.use('Agg')  # Força o uso do backend 'Agg'

ni_img = nib.load(path_nii)
print(ni_img.get_fdata().dtype)
img4D = ni_img.get_fdata().astype(np.uint16)
ni_plot = img4D
voxel_size = ni_img.header.get_zooms()[0:2]

print(f"img shape original: {img4D.shape}")

img4D = np.transpose(img4D,[3,2,0,1])

print(f"img shape corrected: {img4D.shape}")

rect1,rect2 = get_ROI_distance_transform(img4D,voxel_size, 1.4)

img4D_ROI = img4D[:,:,rect1[0]:rect2[0],rect1[1]:rect2[1]]

# img4D_ROI = pad_or_crop_volume(img4D_ROI.shape[3], (128,128,16))
print(f"img shape ROI: {img4D_ROI.shape}")
img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
print(f"img shape retransposed: {img4D_ROI.shape}")

f, axarr = plt.subplots(1,2) 

# Plota as imagens
f, axarr = plt.subplots(1, 2, figsize=(12, 6))  # Tamanho ajustado para visualização clara

# Seleciona fatias específicas para exibição
slice_idx = 4  # Índice da fatia a ser mostrada
axarr[0].imshow(ni_plot[:, :, slice_idx, 3], cmap="gray")  # Visualiza o volume original
axarr[0].set_title("Imagem Original (4D .nii)")
axarr[0].axis("off")  # Remove eixos para clareza

axarr[1].imshow(img4D_ROI[:, :, slice_idx, 3], cmap="gray")  # Visualiza a ROI
axarr[1].set_title("ROI Processada")
axarr[1].axis("off")  # Remove eixos para clareza

# Salva a figura como imagem
plt.savefig(OUTPUT_PATH + "kaggleRoiComparison.jpg", dpi=300)
print(f"Imagem salva em {OUTPUT_PATH}kaggleRoiComparison.jpg") 