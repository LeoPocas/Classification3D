import numpy as np
import cv2
from skimage import filters
from scipy.ndimage import binary_dilation,binary_erosion,distance_transform_edt,binary_fill_holes

def get_gaussian2(width, height):
    x, y = np.meshgrid(np.linspace(-1,1,height), np.linspace(-1,1,width))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return  g

def get_binary_mask(img4D,voxel_size):
    '''
    get a binary mask similar to cocosco's method https://doi.org/10.1016/j.ics.2004.03.179
    return the sub image with all pixel outside the mask as 0 and the binary mask
    '''
    sub = np.std(img4D,axis=0)
    sub = np.max(sub,axis=0)
    size_gaussian = int(16/voxel_size[0])
    if size_gaussian %2 == 0:
        size_gaussian += 1
    
    percentile = np.percentile(sub,95)
    sub[sub>percentile] = percentile
    sub *= get_gaussian2(sub.shape[0],sub.shape[1])
    
    sub /= np.max(np.abs(sub))+np.finfo(float).eps
    sub_blur = np.copy(sub)
    sub_blur = cv2.GaussianBlur(sub_blur,(size_gaussian,size_gaussian),1)
    
    bin_img = np.copy(sub_blur)
    thres = filters.threshold_otsu(bin_img)
    bin_img[bin_img > thres] = 1
    bin_img[bin_img <= thres] = 0

    bin_img2 = bin_img
    
    bin_img2 = binary_erosion(bin_img2,np.array([[0,1,0],[1,1,1],[0,1,0]]),iterations=int(10/voxel_size[0]))
    _, labels_im = cv2.connectedComponents(np.uint8(bin_img2))
    unique, counts = np.unique(labels_im, return_counts=True)
    counts[0] = 0
    if len(unique) > 1:
        max_label = unique[np.argmax(counts)]
    else:
        print("Nenhuma região conectada detectada!")
    
    bin_img2[labels_im != unique[np.argmax(counts)]] = 0
    
    bin_img2 = binary_dilation(bin_img2,np.array([[0,1,0],[1,1,1],[0,1,0]]),iterations=int(10/voxel_size[0]))

    sub[bin_img2 == 0] = 0
    return sub, bin_img2

def get_bounding_box2(img_shape,center,voxel_size,zoom_factor):
    radius = min(round(90/zoom_factor/voxel_size[0]),center[0],center[1],(img_shape-center)[0],(img_shape-center)[1])
    cmin = center-radius
    cmax = center+radius
    return cmin.astype(np.int16),cmax.astype(np.int16)

def get_ROI_distance_transform(img4D,voxel_size, zoom_factor):
    sub,bin_img2 = get_binary_mask(img4D,voxel_size)
    bin_img2 = binary_dilation(bin_img2,np.array([[0,1,0],[1,1,1],[0,1,0]]),iterations=int(10/voxel_size[0]))
    distance_matrix = binary_fill_holes(bin_img2)
    distance_matrix = distance_transform_edt(distance_matrix)
    distance_matrix /= np.max(distance_matrix)
    #distance_matrix[distance_matrix < 0.90] = 0
    center = np.unravel_index(np.argmax(distance_matrix),distance_matrix.shape)
    rect1, rect2 = get_bounding_box2(img_shape = sub.shape,center=np.array([center[0],center[1]]),voxel_size=voxel_size, zoom_factor=zoom_factor)
    return rect1, rect2