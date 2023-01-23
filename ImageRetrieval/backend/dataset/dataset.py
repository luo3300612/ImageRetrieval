import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .default_settings import imagenet_preprocess
import cv2
import numpy as np
from PIL import Image


class ImageRetrievalDataset(Dataset):  # 可以抽象成BaseDataset，但没必要
    def __init__(self, ann_file, preprocess=None):
        self.ann_file = ann_file
        self.image_paths = self.load_anns(ann_file)

        if preprocess is None:
            self.preprocess = imagenet_preprocess
        else:
            self.preprocess = preprocess

    def load_anns(self, ann_file):
        image_paths = []
        with open(ann_file) as f:
            for line in f:
                image_paths.append(line.strip())
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.load_image(self.image_paths[index])
        image_tensor = self.preprocess(image)
        return image_tensor

    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except IOError:
            image_array = np.zeros((224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
        return image
