# 写相关的验证的相关代码  调用预训练好的模型权重
import torch
from dataset import GANDataset
from utils import save_checkpoint, load_checkpoint
from utils import ImagePool
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config

class test(Dataset):
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

def main():
    dataset = test(
        root1=config.TRAIN_DIR+config.TRAINMODE+"/testA", 
        root2=config.TRAIN_DIR+config.TRAINMODE+"/testB", 
        transform=config.transforms
    )


    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,)
    load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,)



    loader = DataLoader(dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True)
    
    loop = tqdm(loader , leave=True)

    for idx,(picture,photo) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        picture = picture.to(config.DEVICE)
        fake_picture = gen_Z(photo)
        fake_photo = gen_H(picture)
        save_image(fake_picture, config.TRAINMODE+"_test"+f"/Gen_picture_{idx}.png")
        save_image(fake_photo, config.TRAINMODE+"_test"+f"/Gen_photo_{idx}.png")


if __name__ == "__main__":
    main()