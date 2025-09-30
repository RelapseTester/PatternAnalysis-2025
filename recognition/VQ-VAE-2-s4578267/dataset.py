import numpy as np
import nibabel as nib
import tqdm as tqdm

import os
import torch
import torch.utils.data.dataloader as dataloader

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


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
    '''

    def __init__(self, X_dir, y_dir, transform=None):
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.transform = transform

        self.data = []

        cdir = os.getcwd()

        X_names = [file for file in os.listdir(self.X_dir) if os.path.isfile(os.path.join(self.X_dir, file))]
        os.chdir(self.X_dir)
        X_images = load_data_2D(X_names)

        os.chdir(cdir)

        y_names = [file for file in os.listdir(self.y_dir) if os.path.isfile(os.path.join(self.y_dir, file))]
        os.chdir(self.y_dir)
        y_images = load_data_2D(y_names)

        os.chdir(cdir)

        for i in range(len(X_images)):
            self.data.append((X_images[i], y_images[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #path = os.path.join(self.X_dir, os.listdir(self.X_dir)[id])
        #image = load_data_2D()

        return None
    

if __name__ == "__main__":

    test_X_dir = "recognition/VQ-VAE-2-s4578267/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
    test_y_dir = "recognition/VQ-VAE-2-s4578267/data/HipMRI_study_complete_release_v1/semantic_labels_anon"

    ds = HipMRIDataset(test_X_dir, test_y_dir)

    print(len(ds))