import numpy as np
import pydicom
import os
import imageio
import math
import nibabel as nib
from scipy.ndimage import zoom
from natsort import natsorted
from pathlib import Path 
from glob import glob
from ast import literal_eval
from tqdm import tqdm
from matplotlib import pyplot as plt
from Classification3D.utils import ROI_PATH, INCOR_PATH, OUTPUT_PATH, DATASETS_PATH, SPACING

def save_gif(img_seq,filename,duration=1,palettesize=256,subrectangles=False):
    '''
    save image sequence as a gif file
    '''
    #img_seq = img_seq-np.min(img_seq)/(np.max(img_seq)-np.min(img_seq)+np.finfo(float).eps)
    volume_normalized = (img_seq - img_seq.min()) / (img_seq.max() - img_seq.min())
    img_seq = (volume_normalized * 255).astype(np.uint8)
    # img_seq = img_seq.astype(np.uint8)
    fps = max(1,int(len(img_seq)/duration))
    imageio.mimsave(filename,img_seq,fps=fps,palettesize=palettesize,subrectangles=subrectangles)

def save_gif_panel(img4D,filename,duration=1,palettesize=256,subrectangles=False):
    '''
    save a img4D (time,slice,x,y,val) in a panel showing each slice side by side. The animation follow the time
    '''
    img  = np.transpose(np.copy(img4D).astype(np.float64),axes=[1,0,2,3])#slice,time,x,y
    nb_slices = img.shape[0]
    height = math.floor(math.sqrt(nb_slices))
    width = math.ceil(nb_slices/height)
    aux = width - (nb_slices % width)
    black_img = np.zeros(shape=[aux,*img.shape[1:]])
    img = np.concatenate([img,black_img],axis=0)
    img_rows = []
    for i in range(0,img.shape[0],width):
        img_rows.append(np.concatenate(img[i:i+width],axis=2))
    panel = np.concatenate(img_rows,axis=1)
    #img_concat = np.concatenate(img,axes=),axis=2)
    save_gif(panel,filename,duration,palettesize,subrectangles)

def get_pixel_resolution_DICOM(dcm):
    xy = dcm['PixelSpacing']
    z = float(dcm["SpacingBetweenSlices"].value) #do not use SliceThickness only, because it does not consider the slice gap. Use SpaceBetweenSlices or calculate the value as 
                                           #the euclidian distance between the 3D point "ImagePositionPatient" of a pair of neighbor slices 
    return [xy[0],xy[1],z]

def get_img4D(path_patient):
    #get all dcm files
    path = os.path.join(path_patient, '**')
    # print(path_patient)
    paths = natsorted(glob(path,recursive=True))
    paths = [path for path in paths if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name)] #and "-" in Path(path).name]

    #print(paths[0])
    dcm = pydicom.dcmread(paths[0])
    #info_dict["voxel_size"] = get_pixel_resolution_DICOM(dcm)
    print(get_pixel_resolution_DICOM(dcm))
    #read all images and group then by slice
    imgs = {}
    for path in paths:
        dcm =  pydicom.dcmread(path)
        if "TriggerTime" in dcm:
            time = int(dcm["TriggerTime"].value) #time instant
        else:
            time = int(dcm["InstanceNumber"].value) #instance number
        #print(path)
        slice = round(dcm["ImagePositionPatient"].value[-1],2) #slice location (do not use SliceLocation since in some cases it swaps base and apex)
        #print(slice,time,dcm["ImagePositionPatient"])
        if slice not in imgs:
            imgs[slice] = {}
        imgs[slice][time] = dcm.pixel_array
    #for k in imgs:
    #    print(k)
    #    print(imgs[k].keys())
    #slice_set = set()
    if "CardiacNumberOfImages" in dcm:
        time_size = dcm["CardiacNumberOfImages"].value
        for k in imgs:
            assert len(imgs[k].keys()) == time_size #make sure the time instants are correct
    else:
        for k in imgs:
            time_size = len(imgs[k].keys())
            break
    #print(imgs.keys())
    '''
    #check if all time instants have images for the same slices
    for time in imgs:
        if not slice_set:
            for k in imgs[time].keys():
                slice_set.add(k)
        else:
            if len(imgs[time]) != len(slice_set):
                print(len(imgs[time]) , len(slice_set))
            assert len(imgs[time]) == len(slice_set)
            for k in imgs[time].keys():
                slice_set.add(k)
            assert len(imgs[time]) == len(slice_set)
    '''
    #sort the slice set (base to apex) and the time instants
    #slice_set = sorted(slice_set)
    img4D = np.zeros(shape=(time_size,len(imgs),dcm.pixel_array.shape[0],dcm.pixel_array.shape[1]),dtype=np.int16)
    #print(img4D.shape)
    for i,slice in enumerate(sorted(imgs.keys(),reverse=True)):
        for j,time in enumerate(sorted(imgs[slice].keys())):
            img4D[j,i] = imgs[slice][time]
    return img4D

def get_ROI_from_path(ROI_path):
    if not ROI_path:
        return {}
    f = open(ROI_path,"r")
    lines = f.read().splitlines() 
    pat = {}
    for line in lines:
        name,ROI = str.split(line,";")
        pat[name] = literal_eval(ROI)
    return pat

def save_Incor4d(img4d, path_patient, pat_name):
    affine = np.eye(4)  # Inicializa uma matriz identidade

    path = os.path.join(path_patient, '**')
    # print(path_patient)
    paths = natsorted(glob(path,recursive=True))
    paths = [path for path in paths if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name)] #and "-" in Path(path).name]

    #print(paths[0])
    dcm = pydicom.dcmread(paths[0])

    # Espaçamento original e novo
    current_spacing = get_pixel_resolution_DICOM(dcm)  # Considera apenas os eixos x, y, z
    new_spacing = SPACING  # 1mm isotrópico nos eixos x e y
    print(f"Reamostrando {pat_name}: Espaçamento atual {current_spacing} -> Novo {new_spacing}")

    # Fatores de escala
    scale_factors = [
        current_spacing[0] / new_spacing[0], 
        current_spacing[1] / new_spacing[1], 
        current_spacing[2] / new_spacing[2],
        1.0  # Dimensão temporal (não muda)
    ]
    print(img4d.shape)
    # Reamostragem
    resampled_data = zoom(img4d, scale_factors, order=3)
    print(resampled_data.shape)
    # Ajuste da matriz affine para refletir a reamostragem
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(1 / np.array(scale_factors[:3]))

    # Cria um novo objeto NIfTI com os dados reamostrados
    resampled_img = nib.Nifti1Image(resampled_data, new_affine)

    # Salva o arquivo reamostrado no diretório de saída
    # nib.save(resampled_img, DATASETS_PATH + '/Incor/' + pat_name)


pat_ignore = ["PN17","PN32","P106","P121","P135","P136","P144","P168","P247","P256","P305","P309","P338","P376","P388","P393","P215","P210","P219","P240","P252"]
pat_erros = []
ROI_locations = get_ROI_from_path(ROI_PATH)
i=0
state = ['Dilatados', 'Normal', 'Hipertrófico']
for path in state:
    pathPat = os.path.join(INCOR_PATH, path)
    for patient in os.listdir(pathPat):
        pat_name = Path(patient).name
        patients = os.path.join(pathPat, patient)
        if pat_name not in pat_ignore:
            try:
                if pat_name in ROI_locations:
                    img4d = get_img4D(patients+'/')
                    # save_gif_panel(img4d, OUTPUT_PATH + pat_name + '.gif')
                    img4d = np.transpose(img4d, [2,3,1,0])
                    save_Incor4d(img4d, patients+'/',path + '/' + pat_name)
            except Exception as e: 
                print(e)
                pat_erros.append(pat_name)
