"""
dataset.py contains HipMRIDataset, a PyTorch Dataset implementation for loading 2D HipMRI images from .nii files.
author: Garrett Bargewell, s4578267
"""

import numpy as np
import nibabel as nib
import tqdm as tqdm

import os
import torch

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def load_HipMRI_slices(imageNames, outDimensions=(256,128), normImage=False, earlyStop=False):
    '''
    Load HipMRI 2d slices.

    imageNames: filenames to load
    normImage: bool (normalise the image 0.0-1.0)
    earlyStop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    #shapes = {}
    images = []
    num = len(imageNames)
    for i, imgName in enumerate(tqdm.tqdm(imageNames)):
        img = nib.load(imgName).get_fdata(caching='unchanged')
        s = img.shape

        # reshape image dimensions
        if s != outDimensions:

            if s[0] > outDimensions[0]:
                h_crop = abs(outDimensions[0] - s[0]) // 2
                img = img[h_crop : s[0] - h_crop,:]
            
            if s[1] > outDimensions[1]:
                w_crop = abs(outDimensions[1] - s[1]) // 2
                img = img[:, w_crop : s[1] - w_crop]

            if s[0] < outDimensions[0] or s[1] < outDimensions[1]:
                h_pad = abs(outDimensions[0] - s[0]) // 2
                w_pad = abs(outDimensions[1] - s[1]) // 2
                img = np.pad(img, pad_width=(h_pad, w_pad), mode='constant', constant_values=0)

        if normImage:
            #fig, axes = plt.subplots(2, figsize=(20, 4))

            #axes[0].imshow(img)
            #print(img.min(), img.max())

            min_val = img.min()
            max_val = img.max()

            img = (img - min_val) / (max_val - min_val)

            #print(img.min(), img.max())

            #axes[1].imshow(img)

            #plt.show()

        img = img[np.newaxis,:,:]
        images.append(img)

        if earlyStop and (num // 10 < len(images)):
            break

    return images
    

# load medical image in 2D
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, earlyStop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    earlyStop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm.tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage
        
        affines.append(affine)
        if i > 20 and earlyStop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images


class HipMRIDataset(torch.utils.data.Dataset):
    '''
    Custom dataset for the HipMRI_complete_release_v1

    X_dir: str, directory to .nii input image files.
    y_dir: str, directory to .nii label image files.
    transform: WIP
    earlyStop: bool, load only the first few data inputs. For testing loading of dataset.
    '''

    def __init__(self, X_dir, transform=None, earlyStop=False):
        self.X_dir = X_dir
        #self.y_dir = y_dir
        self.transform = transform

        self.data = []

        cdir = os.getcwd()

        X_names = [file for file in os.listdir(self.X_dir) if os.path.isfile(os.path.join(self.X_dir, file))]
        os.chdir(self.X_dir)

        X_images = load_HipMRI_slices(X_names, normImage=True, earlyStop=earlyStop)
        #X_images = load_data_2D(X_names, earlyStop=earlyStop)

        os.chdir(cdir)

        #y_names = [file for file in os.listdir(self.y_dir) if os.path.isfile(os.path.join(self.y_dir, file))]
        #os.chdir(self.y_dir)
        #y_images = load_data_2D(y_names, earlyStop=earlyStop)

        #os.chdir(cdir)

        for i in range(len(X_images)):
        #    #self.data.append((X_images[i], y_images[i]))
            self.data.append(X_images[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        #path = os.path.join(self.X_dir, os.listdir(self.X_dir)[id])
        #image = load_data_2D()
        if self.transform:
            return self.transform(self.data[index])
        return self.data[index]
    

if __name__ == "__main__":

    test_X_dir = "recognition/VQ-VAE-2-s4578267/data/keras_slices_data/keras_slices_train"
    #test_y_dir = "recognition/VQ-VAE-2-s4578267/data/keras_slices_data/keras_slices_validate"

    ds = HipMRIDataset(test_X_dir, earlyStop=True)
    dl = torch.utils.data.DataLoader(ds, 64, False)
    

    #for i, (img, label) in enumerate(dl):
    #    print("X data shape:", img.shape)
    #    print("y label shape:", label.shape)
    #    break
    print(ds[0].shape)

    print("Datset length:", len(ds))
    print("Dataloader length:", len(dl))