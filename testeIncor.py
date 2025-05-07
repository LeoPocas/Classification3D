import numpy as np
import pydicom
import os
import imageio
import math
import nibabel as nib
from natsort import natsorted
from pathlib import Path 
from glob import glob
from ast import literal_eval
from tqdm import tqdm
from matplotlib import pyplot as plt
from Classification3D.utils import ROI_PATH, INCOR_PATH, OUTPUT_PATH, INCOR_RESAMPLED_TESTING_PATH
from Classification3D.preprocessing.loadIncor import load_incor_data

# path = INCOR_RESAMPLED_TESTING_PATH+'Normal/PN1.nii'
# path2 = INCOR_PATH + 'Normal/PN1/'

# ni_img = nib.load(path)
# print(ni_img.shape)
# print(ni_img.header.get_zooms())

volumes, label = load_incor_data(training=False)


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


# def save_gif(img_seq,filename,duration=1,palettesize=256,subrectangles=False):
#     '''
#     save image sequence as a gif file
#     '''
#     #img_seq = img_seq-np.min(img_seq)/(np.max(img_seq)-np.min(img_seq)+np.finfo(float).eps)
#     volume_normalized = (img_seq - img_seq.min()) / (img_seq.max() - img_seq.min())
#     img_seq = (volume_normalized * 255).astype(np.uint8)
#     # img_seq = img_seq.astype(np.uint8)
#     fps = max(1,int(len(img_seq)/duration))
#     imageio.mimsave(filename,img_seq,fps=fps,palettesize=palettesize,subrectangles=subrectangles)

# def save_gif_panel(img4D,filename,duration=1,palettesize=256,subrectangles=False):
#     '''
#     save a img4D (time,slice,x,y,val) in a panel showing each slice side by side. The animation follow the time
#     '''
#     img  = np.transpose(np.copy(img4D).astype(np.float64),axes=[1,0,2,3])#slice,time,x,y
#     nb_slices = img.shape[0]
#     height = math.floor(math.sqrt(nb_slices))
#     width = math.ceil(nb_slices/height)
#     aux = width - (nb_slices % width)
#     black_img = np.zeros(shape=[aux,*img.shape[1:]])
#     img = np.concatenate([img,black_img],axis=0)
#     img_rows = []
#     for i in range(0,img.shape[0],width):
#         img_rows.append(np.concatenate(img[i:i+width],axis=2))
#     panel = np.concatenate(img_rows,axis=1)
#     #img_concat = np.concatenate(img,axes=),axis=2)
#     save_gif(panel,filename,duration,palettesize,subrectangles)

# def get_pixel_resolution_DICOM(dcm):
#     xy = dcm['PixelSpacing']
#     z = float(dcm["SpacingBetweenSlices"].value) #do not use SliceThickness only, because it does not consider the slice gap. Use SpaceBetweenSlices or calculate the value as 
#                                            #the euclidian distance between the 3D point "ImagePositionPatient" of a pair of neighbor slices 
#     return [xy[0],xy[1],z]

# def get_img4D(path_patient):
#     #get all dcm files
#     path = os.path.join(path_patient, '**')
#     # print(path_patient)
#     paths = natsorted(glob(path,recursive=True))
#     paths = [path for path in paths if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name)] #and "-" in Path(path).name]

#     #print(paths[0])
#     dcm = pydicom.dcmread(paths[0])
#     #info_dict["voxel_size"] = get_pixel_resolution_DICOM(dcm)
#     print(get_pixel_resolution_DICOM(dcm))
#     #read all images and group then by slice
#     imgs = {}
#     for path in paths:
#         dcm =  pydicom.dcmread(path)
#         if "TriggerTime" in dcm:
#             time = int(dcm["TriggerTime"].value) #time instant
#         else:
#             time = int(dcm["InstanceNumber"].value) #instance number
#         #print(path)
#         slice = round(dcm["ImagePositionPatient"].value[-1],2) #slice location (do not use SliceLocation since in some cases it swaps base and apex)
#         #print(slice,time,dcm["ImagePositionPatient"])
#         if slice not in imgs:
#             imgs[slice] = {}
#         imgs[slice][time] = dcm.pixel_array
#     #for k in imgs:
#     #    print(k)
#     #    print(imgs[k].keys())
#     #slice_set = set()
#     if "CardiacNumberOfImages" in dcm:
#         time_size = dcm["CardiacNumberOfImages"].value
#         for k in imgs:
#             assert len(imgs[k].keys()) == time_size #make sure the time instants are correct
#     else:
#         for k in imgs:
#             time_size = len(imgs[k].keys())
#             break
#     #print(imgs.keys())
#     '''
#     #check if all time instants have images for the same slices
#     for time in imgs:
#         if not slice_set:
#             for k in imgs[time].keys():
#                 slice_set.add(k)
#         else:
#             if len(imgs[time]) != len(slice_set):
#                 print(len(imgs[time]) , len(slice_set))
#             assert len(imgs[time]) == len(slice_set)
#             for k in imgs[time].keys():
#                 slice_set.add(k)
#             assert len(imgs[time]) == len(slice_set)
#     '''
#     #sort the slice set (base to apex) and the time instants
#     #slice_set = sorted(slice_set)
#     img4D = np.zeros(shape=(time_size,len(imgs),dcm.pixel_array.shape[0],dcm.pixel_array.shape[1]),dtype=np.int16)
#     #print(img4D.shape)
#     for i,slice in enumerate(sorted(imgs.keys(),reverse=True)):
#         for j,time in enumerate(sorted(imgs[slice].keys())):
#             img4D[j,i] = imgs[slice][time]
#     return img4D

# def get_ROI_from_path(ROI_path):
#     if not ROI_path:
#         return {}
#     f = open(ROI_path,"r")
#     lines = f.read().splitlines() 
#     pat = {}
#     for line in lines:
#         name,ROI = str.split(line,";")
#         pat[name] = literal_eval(ROI)
#     return pat

# pat_ignore = ["PN17","PN32","P106","P121","P135","P136","P144","P168","P247","P256","P305","P309","P338","P376","P388","P393","P215","P210","P219","P240","P252"]
# pat_erros = []
# ROI_locations = get_ROI_from_path(ROI_PATH)
# i=0
# #for patient in tqdm(valid_paths, desc='Validation'):
# for patient in os.listdir(INCOR_PATH):
#     pat_name = Path(patient).name
#     patients = os.path.join(INCOR_PATH, patient)
#     if pat_name not in pat_ignore:
#         try:
#             if pat_name in ROI_locations:
#                 img4d = get_img4D(patients+'/')
#                 save_gif_panel(img4d, OUTPUT_PATH + pat_name + '.gif')
#                 img4d = np.transpose(img4d, [2,3,1,0])
#                 # plt.imshow(img4d[:,:, 4, 9], cmap='gray')
#                 # plt.savefig(OUTPUT_PATH + pat_name)
#         except Exception as e: 
#             print(e)
#             pat_erros.append(pat_name)