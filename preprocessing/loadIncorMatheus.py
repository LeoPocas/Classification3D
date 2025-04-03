import os
from glob import glob
import h5py
import numpy as np
from natsort import natsorted
import pydicom
from skimage.draw import line
import imageio
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path 
from scipy import io
from scipy.ndimage.morphology import binary_fill_holes
import math
import cv2
from ast import literal_eval

def get_pixel_resolution_DICOM(dcm):
    xy = dcm['PixelSpacing']
    z = float(dcm["SpacingBetweenSlices"].value) #do not use SliceThickness only, because it does not consider the slice gap. Use SpaceBetweenSlices or calculate the value as 
                                           #the euclidian distance between the 3D point "ImagePositionPatient" of a pair of neighbor slices 
    return [xy[0],xy[1],z]

def get_img4D(path_patient,info_dict):
    #get all dcm files
    path = os.path.join(path_patient, '**')
    #print(path_patient)
    paths = natsorted(glob(path,recursive=True))
    paths = [path for path in paths if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name)] #and "-" in Path(path).name]
    #print(paths[0])
    dcm = pydicom.dcmread(paths[0])
    info_dict["voxel_size"] = get_pixel_resolution_DICOM(dcm)
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

def get_segmentations(path_patient, img4D, info_dict,ROI):
    seg4D = np.zeros(shape=(*img4D.shape,3),dtype=np.dtype(bool))
    #load .mat file (make sure there is only one inside the folder)
    path = os.path.join(path_patient, '*.mat')
    paths = natsorted(glob(path,recursive=True))
    assert len(paths)==1
    mat_file = io.loadmat(paths[0])
    #the 4D image (ROI cropped) and the contours coordinates are inside the "setstruct" key
    data = mat_file["setstruct"][0][0]
    img_ROI = np.transpose(data[0],[2,3,0,1]) #data[0] stores the ROI image (transpose to make in order (time, slice, x, y))
    if img_ROI.shape[0] != img4D.shape[0] or img_ROI.shape[1] != img4D.shape[1]:
        print(img_ROI.shape,img4D.shape)
    assert img_ROI.shape[0] == img4D.shape[0] and img_ROI.shape[1] == img4D.shape[1]
    #the var data is a list of np arrays. The contours are the elements that have coordinates/shape = (coordinate, time instant, slice).
    #the first element found that has this shape contains all the x coordinates of the contour points, the second contains all y coordinates. The order is endo, epi
    coord = []
    for i in range(len(data)):
        if len(data[i].shape)==3 and data[i].shape[1]==img_ROI.shape[0] and data[i].shape[2]==img_ROI.shape[1]:
            coord.append(np.transpose(data[i],axes=[1,2,0]))
    assert len(coord) == 4
    x_en,y_en,x_ep,y_ep = coord
    #discover the apex and base slices (so that any slice between then have segmentations)
    apex,base = [x_en.shape[1]-1,0]
    for i in range(x_en.shape[1]):
        if math.isnan(x_en[0,i,0]):
            base += 1
        else:
            break
    for i in range(x_en.shape[1]):
        if math.isnan(x_en[0,x_en.shape[1]-1-i,0]):
            apex -= 1
        else:
            break
    info_dict["apex_slice"] = apex
    info_dict["base_slice"] = base
    if ROI:
        roi_position = ROI
    else:
        #finds the most likely ROI position in the original image by brute force and then produces the segmentation for the entire images
        roi_position = get_ROI_position(img_ROI,img4D)
        print("Não achou ROI. ESTRANHO D:")
    #print(path_patient,roi_position)
    x_en += roi_position[0]
    y_en += roi_position[1]
    x_ep += roi_position[0]
    y_ep += roi_position[1]
    #from the read contours, produce masks for endo and epi and combine then to obtain the ROI segmentation 
    gt_endo = get_gt4D(img4D.shape,x_en,y_en)
    gt_epi = get_gt4D(img4D.shape,x_ep,y_ep)
    seg4D[gt_endo==1] = [0,0,1]
    seg4D[np.logical_and(gt_epi==1,gt_endo==0)] = [0,1,0]
    seg4D[np.logical_and(gt_endo==0, gt_epi==0)] = [1,0,0]
    return seg4D

#the method is VERY SLOW. Only use if you don't know the ROI location. For pat 201, use only one slice
def get_ROI_position(img_ROI,img4D):
    '''
    get the pair (x,y) that minimizes que quadradic difference (expected ROI position)
    '''
    min_x,min_y,min_score = [0,0,9999999999999]
    x_size,y_size = img_ROI.shape[2:4]
    slices = img_ROI.shape[1]
    img_eq = image_histogram_equalization(img_ROI[0])
    for x in range(0,img4D.shape[2]-x_size):
        for y in range(0,img4D.shape[3]-y_size):
            cut = img4D[0,:,x:x+x_size,y:y+y_size]
            score = get_score(cut,img_eq)
            if score < min_score:
                min_score,min_x,min_y = [score,x,y]
    return (min_x,min_y)
                
def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram (if bins < number of different intensities, the bins array represents the quantization values of the image)
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=False)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized *= (number_bins-1)
    return image_equalized.reshape(image.shape)

def get_score(img1,img2_eq):
    img1 = image_histogram_equalization(img1)
    return np.sum((img2_eq-img1)**2)

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
       

def get_gt4D(img_shape,x_coord,y_coord):
    '''
    draw contours by linear interpolation of coordinates and fills the resulting polygon 
    '''
    gt = np.zeros(shape=img_shape)
    for time in range(img_shape[0]):
        for slice in range(img_shape[1]):
            if math.isnan(x_coord[time,slice,0]):
                continue
            for i in range(-1,x_coord.shape[2]-1):
                x1,y1 = np.array([min(x_coord[time,slice,i],img_shape[2]-1),min(y_coord[time,slice,i],img_shape[3]-1)],dtype=np.int32)
                x2,y2 = np.array([min(x_coord[time,slice,i+1],img_shape[2]-1),min(y_coord[time,slice,i+1],img_shape[3]-1)],dtype=np.int32)
                lin = line(x1,y1,x2,y2)
                a = gt[time,slice]
                a[lin] = 1
            gt[time,slice] = binary_fill_holes(gt[time,slice])
    return gt

def create_group(group,patient_path,ROI=None):
    
    patient_name = Path(patient_path).name
    patient_group = group.create_group(patient_name)
    patient_group.attrs["Group"] = Path(patient_path).parent.name
    img4D = get_img4D(patient_path+"/",patient_group.attrs)
    gt4D = get_segmentations(patient_path,img4D,patient_group.attrs,ROI)
    #find ED and ES phases by calculating the maximum and minimum volume instants
    max_vol,min_vol = [-1,999999999]
    ED_phase,ES_phase = [0,0]
    for time in range(gt4D.shape[0]):
        vol = np.sum(gt4D[time,...,2])
        if vol > max_vol:
            max_vol = vol
            ED_phase = time
        if vol < min_vol:
            min_vol = vol
            ES_phase = time
    patient_group.attrs["ED_phase"] = ED_phase
    patient_group.attrs["ES_phase"] = ES_phase
    patient_group.create_dataset('img',data=img4D)
    patient_group.create_dataset('gt',data=gt4D)
     


def generate_dataset(path_dataset, name="incor.hdf5",ROI_path=None,valid_prop=0.25):
    '''
    path_dataset = path that contains folders 'Normal', 'Hipertrófico' and 'Dilatados'
    '''
    pat_ignore = ["PN17","PN32","P106","P121","P135","P136","P144","P168","P247","P256","P305","P309","P338","P376","P388","P393","P215","P210","P219","P240","P252"]
    pat_erros = []
    patologies = {"Normal":list(),"Hipertrófico":list(),"Dilatados":list()}
    rng = np.random.default_rng()
    for k in patologies:
        patologies[k] = np.array([f.path for f in os.scandir(path_dataset+"/"+k) if f.is_dir() and k in f.path])
        rng.shuffle(patologies[k])

    valid_paths = np.concatenate([patologies[k][:int(patologies[k].shape[0]*valid_prop)] for k in patologies])
    train_paths = np.concatenate([patologies[k][int(patologies[k].shape[0]*valid_prop):] for k in patologies])

    ROI_locations = get_ROI_from_path(ROI_path)

    with h5py.File(name, "w") as h5f:

        # Training samples ###
        group = h5f.create_group("train")
        for patient in tqdm(train_paths, desc='Training'):
            pat_name = Path(patient).name
            if pat_name not in pat_ignore:
                try:
                    if pat_name in ROI_locations:
                        create_group(group,patient,ROI_locations[pat_name])
                    else:
                        create_group(group,patient)
                except Exception as e: 
                    print(e)
                    pat_erros.append(pat_name)
       

        # Validation samples ###
        group = h5f.create_group("valid")
        for patient in tqdm(valid_paths, desc='Validation'):
            pat_name = Path(patient).name
            if pat_name not in pat_ignore:
                try:
                    if pat_name in ROI_locations:
                        create_group(group,patient,ROI_locations[pat_name])
                    else:
                       create_group(group,patient)
                except Exception as e: 
                    print(e)
                    pat_erros.append(pat_name)
    print(pat_erros)

def save_gif(img_seq,filename,duration=1,palettesize=256,subrectangles=False):
    '''
    save image sequence as a gif file
    '''
    #img_seq = img_seq-np.min(img_seq)/(np.max(img_seq)-np.min(img_seq)+np.finfo(float).eps)
    img_seq = img_seq.astype(np.uint8)
    fps = max(1,int(len(img_seq)/duration))
    imageio.mimsave(filename,img_seq,fps=fps,palettesize=palettesize,subrectangles=subrectangles)

def save_gif_panel(img4D,filename,duration=1,palettesize =256,subrectangles=False):
    '''
    save a img4D (time,slice,x,y,val) in a panel showing each slice side by side. The animation follow the time
    '''
    img  = np.transpose(np.copy(img4D).astype(np.float),axes=[1,0,2,3,4])#slice,time,x,y,val
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
    save_gif(panel,filename,duration,palettesize ,subrectangles)

def main():
    falty = []
    p = "D:/datasets_RMC/datasets_hdf5/incor.hdf5"
    with h5py.File(p,"r") as d:
        for group in d:
            for pat in tqdm(d[group],desc="PAT"):
                try:
                    gt  = np.copy(d[group][pat]["gt"]).astype(np.float)
                    img  = np.copy(d[group][pat]["img"]).astype(np.float)
                    img = img[...,np.newaxis]*[255,255,255]/np.max(img)
                    arg_max = np.argmax(gt,axis=-1)
                    gt[arg_max==0] = img[arg_max==0]
                    gt[arg_max==1] = [255,215,0]#amarelo
                    gt[arg_max==2] = [30,144,255]#azul
                    gt_resized = np.zeros(shape=[gt.shape[0],gt.shape[1],gt.shape[2]//2,gt.shape[3]//2,3])
                    for i in range(gt.shape[0]):
                        for j in range(gt.shape[1]):
                            gt_resized[i,j] = cv2.resize(gt[i,j], dsize=(gt_resized.shape[3],gt_resized.shape[2]))
                    save_gif_panel(gt_resized,"D:/datasets_RMC/gifs/incor_gifs2/{}.gif".format(pat),palettesize=128,subrectangles=False)
                except Exception as e: 
                    print(e)
                    falty.append(pat)
    print(falty)
        #print(img_concat.shape)
        
if __name__ == "__main__":
    generate_dataset(path_dataset="D:/datasets_RMC/dataset incor",ROI_path="D:/datasets_RMC/dataset incor/ROI_locations.txt",name="D:/datasets_RMC/datasets_hdf5/incor.hdf5")
    main()


