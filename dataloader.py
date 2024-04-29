import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import options

'''
Dataset/Dataloader: ensure train/val split
'''

def _get_image_filepaths(dirpath):
    image_filepaths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_filepaths.append(os.path.join(root, file))
    return image_filepaths

class ImgAttrDataset(Dataset):
    def __init__(self, image_dirpath, imagelist_filepath, feature_df, transforms):
        assert(os.path.exists(image_dirpath))
        assert(transforms != None)

        self.image_dirpath = image_dirpath
        # self.image_paths = _get_image_filepaths(image_dirpath)
        self.image_paths = list()
        
        with open(imagelist_filepath, "r") as file:
            for line in file:
                lines = line.strip()
                lines = lines.split(" ")
                self.image_paths.append(lines[1])

        self.transforms = transforms
        self.features = np.asarray(feature_df.values, dtype=float)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(os.path.join(self.image_dirpath, self.image_paths[idx]))
        except:
            raise ValueError(os.path.join(self.image_dirpath, self.image_paths[idx]))

        image = self.transforms(image)
        features = self.features[idx]
        
        return image, features
           
def get_dataloader(img_dirpath, imglist_filepath, feature_df, batch_size=None):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImgAttrDataset(img_dirpath, imglist_filepath, feature_df, data_transforms)
    dataloader = DataLoader(dataset, batch_size = options.LOADER_BATCH_SIZE if not batch_size else batch_size, shuffle=True)
    
    return dataloader