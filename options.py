import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""
GLOBAL Filepaths
"""
CUB_DIRPATH = "CUB_200_2011/CUB_200_2011/"
IMG_DIRPATH = "images-processed"
IMGLIST_FILEPATH = os.path.join(CUB_DIRPATH, "images.txt")

"""
Dataset/Dataloader
"""
LOADER_BATCH_SIZE = 16