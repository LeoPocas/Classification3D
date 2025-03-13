import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
import nibabel as nib
from preprocessing.load_data import load_4d_roi_sep, load_acdc_data_3d
from preprocessing.load_mms import load_mmms_data
from utils import OUTPUT_PATH
path_nii = './ACDC/database/training/patient057/patient057_4d.nii.gz'
matplotlib.use('Agg')  # Força o uso do backend 'Agg'

ni_img = nib.load(path_nii)
img4D = ni_img.get_fdata().astype(np.uint16)
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

# axarr[0].imshow(img4D[1,1],cmap="gray")
# axarr[1].imshow(img4D_ROI[:,:,1,1],cmap="gray")

plt.savefig(OUTPUT_PATH + "algo.jpg")

# volumes, label, discard = load_acdc_data_3d()
volumes, label, discard = load_4d_roi_sep()

# Verificar o shape do primeiro volume
print(f"Shape do primeiro volume: {volumes[56].shape}")
print(f"Volumes: {volumes.shape}, \n Labels:{label}")
# Selecionar uma fatia específica do volume para visualização
# Aqui, escolhemos a primeira profundidade e a primeira fatia de tempo
fatia_altura_largura = volumes[56, :, :, 5, 0]

# Exibir a fatia selecionada
plt.imshow(fatia_altura_largura, cmap="gray")
plt.title("Fatia do Volume")
plt.colorbar()
plt.savefig(OUTPUT_PATH + "volume_slice.png")

volume_index = 56  
slice_index = volumes.shape[3] // 2  # Fatia central

# Plotagem da imagem do volume
plt.figure(figsize=(10, 5))
plt.imshow(volumes[volume_index, :, :, slice_index, 0], cmap='gray')
plt.title(f"Volume {volume_index}, Slice {slice_index}")
plt.axis('off')
plt.savefig(OUTPUT_PATH + "/new.png")

volume_index = 0  # Escolha o volume que deseja visualizar

# Itera sobre todos os slices na quarta dimensão (eixo de tempo)
# for i in range(volumes.shape[3]):
#     plt.figure(figsize=(10, 5))
#     plt.imshow(volumes[volume_index, :, :, i, 0], cmap='gray')
#     plt.title(f"Volume {volume_index}, Slice {i}")
#     plt.axis('off')
#     plt.savefig(str(i) + "_new.png")
#     plt.show()

volumes, label, discard = load_mmms_data()

# Verificar o shape do primeiro volume
print(f"Shape do primeiro volume: {volumes[56].shape}")
print(f"Volumes: {volumes.shape}, \n Labels:{label}")
# Selecionar uma fatia específica do volume para visualização
# Aqui, escolhemos a primeira profundidade e a primeira fatia de tempo
fatia_altura_largura = volumes[56, :, :, 5, 0] 

# Exibir a fatia selecionada
plt.imshow(fatia_altura_largura, cmap="gray")
plt.title("Fatia do Volume")
plt.colorbar()
plt.savefig(OUTPUT_PATH + "volume_slicemm.png")

volume_index = 56  
slice_index = volumes.shape[3] // 2  # Fatia central

# Plotagem da imagem do volume
plt.figure(figsize=(10, 5))
plt.imshow(volumes[volume_index, :, :, slice_index, 0], cmap='gray')
plt.title(f"Volume {volume_index}, Slice {slice_index}")
plt.axis('off')
plt.savefig(OUTPUT_PATH + "/newmm.png")

volume_index = 0  # Escolha o volume que deseja visualizar