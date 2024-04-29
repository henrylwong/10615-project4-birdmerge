import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""
Logging
"""
LOGGING_FILEPATH = "main.log"

"""
GLOBAL Filepaths
"""
CUB_DIRPATH = "CUB_200_2011/CUB_200_2011/"
IMG_DIRPATH = os.path.join(CUB_DIRPATH, "images")
BBOX_FILEPATH = os.path.join(CUB_DIRPATH, "bounding_boxes.txt")
IMGLIST_FILEPATH = os.path.join(CUB_DIRPATH, "images.txt")
ATTRIBUTES_FILEPATH = os.path.join(CUB_DIRPATH, "attributes", "attributes.txt")
IMG_ATTRIBUTES_FILEPATH = os.path.join(CUB_DIRPATH, "attributes", "image_attribute_labels.txt")

IMG_PROCESSED_DIRPATH = "images-processed"
IMGLIST_PROCESSED_FILEPATH = "images-processed.txt"
FEATURES_DF_PKL_FILEPATH = 'features_df.pkl'

SAVES_DIRPATH = "saves"

"""
Dataset/Dataloader
"""
LOADER_BATCH_SIZE = 16

"""
Model
"""
IMG_CHANNELS = 3
HIDDEN_DIM = 512
LATENT_DIM = 64 # @henry: 29 attributes to be encoded
ENCODER_CHANNELS = (16, 32, 64, 128, 32)
DECODER_CHANNELS = (128, 64, 32, 16, 8)
ENCODER_CHANNELS = (64, 128, 256, 512, 32)
DECODER_CHANNELS = (512, 256, 128, 64, 32)

NUM_EPOCHS = 1000
OPT_LR = 0.001

DO_AR_LOSS = True
DO_L1_REG = True

"""
Training Params
"""
RECON_WEIGHT = 1.0
BETA = 2.0
GAMMA = 10.0
AR_FACTOR = 100.0
L1_REG_FACTOR = 0.00005