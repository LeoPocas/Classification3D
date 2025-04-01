import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import nibabel as nib
from utils import OUTPUT_PATH
from scipy.ndimage import zoom
from Classification3D.preprocessing.load_mms import load_mms_data, load_mms_data_pure
from Classification3D.preprocessing.equalizacao import apply_clahe, pad_or_crop_volume
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.utils import *

path_nii2 = OUTPUT_PATH + 'mms_resampled/Training/B8J7R4/B8J7R4_sa.nii.gz'
path_nii = MMs_PATH + 'Training/Labeled/B8J7R4/B8J7R4_sa.nii.gz'
matplotlib.use('Agg')  # Força o uso do backend 'Agg'

ni_img = nib.load(path_nii)
print(ni_img.shape)

img4D = ni_img.get_fdata().astype(np.float64)
voxel_size = ni_img.header.get_zooms()[0:3]

ni_img2 = nib.load(path_nii2)
voxel_size2 = ni_img2.header.get_zooms()[0:3]

print(voxel_size, voxel_size2)
data = ni_img.get_fdata()
# voxel_size = ni_img.header.get_zooms()[0:2]

img4D = np.transpose(img4D,[3,2,0,1])


rect1,rect2 = get_ROI_distance_transform(img4D,voxel_size, 1.0)

img4D_ROI = img4D[:,:,rect1[0]:rect2[0],rect1[1]:rect2[1]]

# img4D_ROI = pad_or_crop_volume(img4D_ROI.shape[3], (128,128,16))
img4D_ROI = np.transpose(img4D_ROI, [2, 3, 1, 0])
print(img4D_ROI.shape)
for t in range(img4D_ROI.shape[3]):                    
    if t == 1:
        volume_3d_ED = img4D_ROI[:, :, :, t]
        volume_3d_ED = apply_clahe(volume_3d_ED)
        volume_3d_ED = pad_or_crop_volume(volume_3d_ED, TARGET_SHAPE)
        volume_3d_ED = np.repeat(volume_3d_ED[..., np.newaxis], 1, axis=-1)
    elif t == 9:
        volume_3d_ES = img4D_ROI[:, :, :, t]
        volume_3d_ES = apply_clahe(volume_3d_ES)
        volume_3d_ES = pad_or_crop_volume(volume_3d_ES, TARGET_SHAPE)
        volume_3d_ES = np.repeat(volume_3d_ES[..., np.newaxis], 1, axis=-1)

# plt.subplot(233)
# plt.imshow(img4D_ROI[:,:, 4, 9], cmap='gray')
# # plt.subplot(234)
# # plt.imshow(img4D_ROI[:,:, 4, 1], cmap='gray')
# # plt.subplot(231)
# # plt.imshow(volume_3d_ED[:,:,4,0], cmap='gray')
# plt.subplot(232)
# plt.imshow(volume_3d_ES[:,:,4,0], cmap='gray')
# plt.savefig("Clahes.png")
affine = ni_img.affine
header = ni_img.header

# current_spacing = header.get_zooms()
# print(current_spacing)

# new_spacing = (1.0,1.0, current_spacing[2])

# scale_factors = [ current_spacing[0] / new_spacing[0],
#                   current_spacing[1] / new_spacing[1],
#                   current_spacing[2] / new_spacing[2],
#                   1.0]

# resampled_data = zoom(data, scale_factors, order=3)

# new_affine = affine.copy()
# new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / np.array(scale_factors[:3]))

# resampled_img = nib.Nifti1Image(resampled_data, new_affine, header)

# print(resampled_img.shape)

original_slice = data[:, :, 4, 12]
# resampled_slice = resampled_data[:, :, 4, 12]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Volume inicial
axes[0].imshow(original_slice , cmap='gray', aspect='auto')
axes[0].set_title('Volume Inicial')
axes[0].axis('off')

# # Volume reamostrado
# axes[1].imshow(resampled_slice, cmap='gray', aspect='auto')
# axes[1].set_title('Volume Reamostrado')
# axes[1].axis('off')

# Salvando a imagem comparativa
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'comparacao_volumes.png', dpi=300)
plt.close()

volumes, label, patient = load_mms_data(training=False)


# # Verificar o shape do primeiro volume
# print(f"Shape do primeiro volume: {volumes[66].shape}")
# print(f"Volumes: {volumes.shape}, \n Labels:{label}")
# # Selecionar uma fatia específica do volume para visualização
# # Aqui, escolhemos a primeira profundidade e a primeira fatia de tempo
# fatia_altura_largura = volumes[66, :, :, 5, 0]

# # Exibir a fatia selecionada
# plt.imshow(fatia_altura_largura, cmap="gray")
# plt.title("Fatia do Volume")
# plt.colorbar()
# plt.savefig(OUTPUT_PATH + "volume_slice.png")

for i in range(volumes.shape[0]):
    volume_index = i  
    slice_index = volumes.shape[3] // 2  # Fatia central
    print(volumes[i,:,:,:,:].shape)
    # Plotagem da imagem do volume
    plt.figure(figsize=(10, 5))
    plt.imshow(volumes[volume_index, :, :, slice_index, 0], cmap='gray')
    plt.title(f"Volume {volume_index}, Slice {slice_index}")
    plt.axis('off')
    plt.savefig(OUTPUT_PATH + f"/roiMMs{i}.png")
# iterator = 0
# for path in os.listdir(MMs_REESPACADO+'Testing/'):
#     path2 = os.path.join(MMs_REESPACADO+'Testing/', path)
#     for file in os.listdir(path2):
#         filepath = os.path.join(path2, file)
#         iterator += 1
#         print(file)
#         # for file in os.listdir(files):
#         if file.endswith('.nii.gz'):
#             ni_img = nib.load(filepath)
#             img4D = ni_img.get_fdata()
#             slice_data = img4D[:, :, 8, 0]
#             normalized_slice = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
#             plt.imshow(normalized_slice, cmap='gray')
#             plt.title(f"Volume {file}, Slice {8}")
#             plt.axis('off')
#             plt.savefig(OUTPUT_PATH + f"/roiMMs{file}.png")
#         if iterator > 10: continue