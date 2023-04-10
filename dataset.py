from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class GANDataset(Dataset):
    def __init__(self, root1, root2, transform=None):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform

        self.images1 = os.listdir(root1)
        self.images2 = os.listdir(root2)
        self.length_dataset = max(len(self.images1), len(self.images2)) # 1000, 1500
        self.len1 = len(self.images1)
        self.len2 = len(self.images2)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img1 = self.images1[index % self.len1]
        img2 = self.images2[index % self.len2]

        path1 = os.path.join(self.root1, img1)
        path2 = os.path.join(self.root2, img2)

        img1 = np.array(Image.open(path1).convert("RGB"))
        img2 = np.array(Image.open(path2).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=img1, image0=img2)
            img1 = augmentations["image"]
            img2 = augmentations["image0"]

        return img1, img2
