import numpy as np
import pydicom
import os
from natsort import natsorted
from pathlib import Path 
from glob import glob
from ast import literal_eval
from tqdm import tqdm
from Classification3D.utils import ROI_PATH, INCOR_PATH

def get_pixel_resolution_DICOM(dcm):
    xy = dcm['PixelSpacing']
    z = float(dcm["SpacingBetweenSlices"].value) #do not use SliceThickness only, because it does not consider the slice gap. Use SpaceBetweenSlices or calculate the value as 
                                           #the euclidian distance between the 3D point "ImagePositionPatient" of a pair of neighbor slices 
    return [xy[0],xy[1],z]

def get_img4D(path_patient):
    #get all dcm files
    path = os.path.join(path_patient, '**')
    #print(path_patient)
    paths = natsorted(glob(path,recursive=True))
    paths = [path for path in paths if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name)] #and "-" in Path(path).name]
    #print(paths[0])
    dcm = pydicom.dcmread(paths[0])
    #info_dict["voxel_size"] = get_pixel_resolution_DICOM(dcm)
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

pat_ignore = ["PN17","PN32","P106","P121","P135","P136","P144","P168","P247","P256","P305","P309","P338","P376","P388","P393","P215","P210","P219","P240","P252"]
pat_erros = []
ROI_locations = get_ROI_from_path(ROI_PATH)

#for patient in tqdm(valid_paths, desc='Validation'):
for patient in tqdm(INCOR_PATH):
    pat_name = Path(patient).name
    if pat_name not in pat_ignore:
        try:
            if pat_name in ROI_locations:
                get_img4D(patient)
        except Exception as e: 
            print(e)
            pat_erros.append(pat_name)