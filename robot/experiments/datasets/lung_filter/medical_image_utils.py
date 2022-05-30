import os
from typing import Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import SimpleITK as sitk
import torch
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans
import torchio as tio


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s, force=True) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
            if sec_num >= len(slices):
                break
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePostionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLoacation - slices[1].SliceLoacation)

    for s in slices:
        s.slice_thickness = slice_thickness

    return slices

def load_IMG(file_path, shape, spacing, new_spacing):
    # reader = sitk.ImageFileReader()
    # reader.SetImageIO("NiftiImageIO")
    # reader.SetFileName(file_path)
    # image = reader.Execute()
    # image_np = sitk.GetArrayFromImage(image)
    
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    # image = win_scale(image, -600, 1200, np.float32, [0, 1])

    # mask, bbox = seg_bg_mask(image, True)

    # for i in range(0,10):
    #     plt.imshow((mask*image)[:,:,i*20])
    #     plt.savefig("./log/image_%i.jpg"%i)
    
    # image = win_scale(image, 490, 820, np.float32, [0, 1])
    # image = image.astype(np.float32)
    # image_max = np.max(image)
    # image_min = np.min(image)
    # image = image/(image_max - image_min)

    # image[image>300] = 0
    # image = (image/1000.0+1)*1.673

    return image #, mask, bbox


def load_ITK(path):
    if path is not None:
        img = sitk.ReadImage(path)
        spacing_sitk = img.GetSpacing()
        img_sz_sitk = img.GetSize()
        origin = img.GetOrigin()
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(img)), np.flipud(spacing_sitk), np.flipud(img_sz_sitk), np.flipud(origin)
    else:
        return None, None, None, None


def resample(imgs, spacing, new_spacing, mode="linear"):
    """
    :return: new_image, true_spacing
    """
    dim = len(imgs.shape)
    if dim == 3 or dim == 2:
        # If the image is 3D or 2D image
        # Use torchio.Resample to resample the image.

        # Create a sitk Image object then load this object to torchio Image object
        imgs_itk = sitk.GetImageFromArray(imgs)
        imgs_itk.SetSpacing(np.flipud(spacing).astype(np.float64))
        imgs_tio = tio.ScalarImage.from_sitk(imgs_itk)
        
        # Resample Image
        resampler = tio.Resample(list(np.flipud(new_spacing)), image_interpolation=mode)
        new_imgs = resampler(imgs_tio).as_sitk()

        # Prepare return value
        new_spacing = new_imgs.GetSpacing()
        new_imgs = sitk.GetArrayFromImage(new_imgs)
        resize_factor = np.array(imgs.shape) / np.array(new_imgs.shape)
        return new_imgs, new_spacing, resize_factor
    elif dim == 4:
        # If the input is a batched 3D image
        # Run resample on each image in the batch.
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice, true_spacing, resize_factor = resample(slice, spacing, new_spacing, mode=mode)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg),[1, 2, 3, 0])
        return newimg, true_spacing, resize_factor
    else:
        raise ValueError('wrong shape')

def smoother(img, sigma=3):
    D = img.shape[0]
    for i in range(D):
        img[i] = gaussian_filter(img[i], sigma)
    return img

def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
  
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)


def seg_bg_mask(img):
    """
    Calculate the segementation mask for the whole body.
    Assume the dimensions are in Superior/inferior, anterior/posterior, right/left order.
    :param img: a 3D image represented in a numpy array.
    :return: The segmentation Mask. BG = 0
    """
    (D,W,H) = img.shape

    img_cp = np.copy(img)
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    mean = np.mean(middle)  

    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # clear bg
    dilation = morphology.dilation(thresh_img,np.ones([4,4,4]))
    eroded = morphology.erosion(dilation,np.ones([4,4,4]))

    # Select the largest area besides the background
    labels = measure.label(eroded, background=1)
    regions = measure.regionprops(labels)
    roi_label = 0
    max_area = 0
    for region in regions:
        if region.label != 0 and region.area > max_area:
            max_area = region.area
            roi_label = region.label
    thresh_img = np.where(labels==roi_label, 1, 0)

    # bound the ROI. 
    # TODO: maybe should check for bounding box
    # thresh_img = 1 - eroded
    sum_over_traverse_plane = np.sum(thresh_img, axis=(1,2))
    top_idx = 0
    for i in range(D):
        if sum_over_traverse_plane[i] > 0:
            top_idx = i
            break
    bottom_idx = D-1
    for i in range(D-1, -1, -1):
        if sum_over_traverse_plane[i] > 0:
            bottom_idx = i
            break
    for i in range(top_idx, bottom_idx+1):
        thresh_img[i]  = morphology.convex_hull_image(thresh_img[i])

    labels = measure.label(thresh_img)
    
    bg_labels = []
    corners = [(0,0,0),(-1,0,0),(0,-1,0),(-1,-1,0),(0,-1,-1),(0,0,-1),(-1,0,-1),(-1,-1,-1)]
    for pos in corners:
        bg_labels.append(labels[pos])
    bg_labels = np.unique(np.array(bg_labels))
    
    mask = labels
    for l in bg_labels:
        mask = np.where(mask==l, -1, mask)
    mask = np.where(mask==-1, 0, 1)

    roi_labels = measure.label(mask, background=0)
    roi_regions = measure.regionprops(roi_labels)
    bbox = [0,0,0,D,W,H]
    for region in roi_regions:
        if region.label == 1:
            bbox = region.bbox
    
    return mask, bbox

def seg_lung_mask(img):
    """
    Calculate the segementation mask either for lung only.
    :param img: a 3D image represented in a numpy array.
    :return: The segmentation Mask.
    """
    (D,W,H) = img.shape

    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    mean = np.mean(middle)  
    img_max = np.max(img)
    img_min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==img_max]=mean
    img[img==img_min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([4,4,4]))
    dilation = morphology.dilation(eroded,np.ones([4,4,4]))

    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_regions = []
    
    for prop in regions:
        B = prop.bbox
        if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/5 and B[4]<W/20*16 and B[1]>W/10 and
                B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20 and B[2]>H/10 and B[5]<H/20*19 and
                B[3]-B[0]>D/4):
            good_regions.append(prop)
            continue
            print(B)
        
        if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/6 and B[4]<W/20*18 and B[1]>W/20 and
                B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20):
            good_regions.append(prop)
            continue
        
        if B[4]-B[1]<W/20*18 and B[4]-B[1]>W/20 and B[4]<W/20*18 and B[1]>W/20:
            good_regions.append(prop)
            continue
    
    # Select the most greatest region
    good_regions = sorted(good_regions, key=lambda x:x.area, reverse=True)
    
    mask = np.ndarray([D,W,H],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    good_labels_bbox = []
    for N in good_regions[:2]:
        mask = mask + np.where(labels==N.label, 1, 0)
        good_labels_bbox.append(N.bbox)
    
    # Get the bbox of lung
    bbox = [D/2, W/2, H/2, D/2, W/2, H/2]
    for b in good_labels_bbox:
        for i in range(0, 3):
            bbox[i] = min(bbox[i], b[i])
            bbox[i+3] = max(bbox[i+3], b[i+3])
    
    mask = morphology.dilation(mask, np.ones([4,4,4])) # one last dilation
    mask = morphology.erosion(mask,np.ones([4,4,4]))

    return mask, bbox

def binary_dilation(img, radius = 1):
    return morphology.binary_dilation(img, morphology.ball(radius))

def normalize_intensity(img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [0,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        return normalized_img

if __name__=="__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dicomPath = '../../Data/Raw/DICOMforMN/S00001/SER00002'
    sdtPath = '../../Data/Raw/NoduleStudyProjections/001/DICOM/'
    processed_file_folder = '../../Data/Preprocessed/sdt0001'
    
    # Processing CT images
    case = load_scan(dicomPath)
    image = np.stack([s.pixel_array for s in case])
    image = image.astype(np.int16)

    for slice_number in range(len(case)):
        intercept = case[slice_number].RescaleIntercept
        slope = case[slice_number].RescaleSlope
            
        # Hounsfield Unit = pixel_value * rescale_slope + rescale_intercept
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)

    case_pixels = np.array(image, dtype=np.float)  # HU value
    spacing = np.array([case[0].SliceThickness, case[0].PixelSpacing[0], case[0].PixelSpacing[1]], dtype=np.float32)  # spacing z,x,y
    case_pixels, new_spacing = resample(case_pixels, spacing, [1, 1, 1])
    case_pixels = np.flip(case_pixels, axis=0).copy() # z is upside down, so flip it

    # case_pixels = win_scale(case_pixels, -650., 100., np.float, [0., 1.])
    # plt.imshow(case_pixels[120], cmap='gray')
    # plt.savefig("./data/case_lung.png")
    # Transform HU to attenuation coefficient u
    case_pixels = (case_pixels/1000.0+1)*1.673
    
    # Save preprocessed CT to numpy
    if not os.path.exists(processed_file_folder):
        os.mkdir(processed_file_folder)
    np.save(processed_file_folder+"/ct.npy", case_pixels)
    
    # Processing raw data from sDT
    image = [dicom.read_file(sdtPath + '/' + s).pixel_array for s in os.listdir(sdtPath)]
    image = np.array(image)
    image = image.astype(np.float32)
    image = image[:, 0:2052]
    image = -np.log(image/65535+0.0001)

    
    # proj_y[proj_y==0]=1
    # proj_y[proj_y>16000] = 16000
    # proj_y = -proj_y
    # # proj_y = np.log(proj_y)
    # # proj_y[proj_y<7] = 7
    # # proj_y = -proj_y
    # proj_y_max = np.max(proj_y, axis=(1,2))
    # proj_y_min = np.min(proj_y, axis=(1,2))
    # dur = proj_y_max - proj_y_min
    # for i in range(proj_y.shape[0]):
    #     proj_y[i] = (proj_y[i] - proj_y_min[i])/dur[i]*255
    np.save(processed_file_folder+"/projection.npy", image)
