import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2  #A是一个数据增强的包

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/"
VAL_DIR = "data/"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE_A = 5
LAMBDA_CYCLE_B = 5
LAMBDA_TV = 0.1
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False     #对预训练好的模型继续进行训练
SAVE_MODEL = True
TRAINMODE = "vangogh2photo"   #choose the train dataset
CHECKPOINT_GEN_H = "model_save/"+TRAINMODE+"_genh.pth.tar"
CHECKPOINT_GEN_Z = "model_save/"+TRAINMODE+"_genz.pth.tar"
CHECKPOINT_CRITIC_H = "model_save/"+TRAINMODE+"_critich.pth.tar"
CHECKPOINT_CRITIC_Z = "model_save/"+TRAINMODE+"_criticz.pth.tar"
POOLSIZE = 50
LR_DECAY = 50


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)