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

def train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    
    #imagepool原文的作者说这样可以减小震荡
    fake2_pool = ImagePool(config.POOLSIZE)
    fake1_pool = ImagePool(config.POOLSIZE)

    for idx, (photo1, photo2) in enumerate(loop):
        photo1 = photo1.to(config.DEVICE)
        photo2 = photo2.to(config.DEVICE)

        # # Train Discriminators H and Z
        
        fake2 = fake2_pool.query(gen_H(photo1))
        D_H_real = disc_H(photo2)          
        D_H_fake = disc_H(fake2.detach())        
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))   #这里并不是直接用一个数字来表示输出的概率，而是希望输出的方格里的值为0~1来表示概率
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss
        H_reals += D_H_real.mean().item()
        H_fakes += D_H_fake.mean().item()

        fake1 = fake1_pool.query(gen_Z(photo2))
        D_Z_real = disc_Z(photo1) 
        D_Z_fake = disc_Z(fake1.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
        D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        # Train Generators H and Z
        # with torch.cuda.amp.autocast():
            # adversarial loss for both generators
        D_H_fake = disc_H(fake2)
        D_Z_fake = disc_Z(fake1)
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

        # cycle loss
        cycle_1 = gen_Z(fake2)
        cycle_2 = gen_H(fake1)
        cycle_1_loss = l1(photo1, cycle_1)
        cycle_2_loss = l1(photo2, cycle_2)


        #降噪算法
        tv_loss = 0.5 * (torch.abs(fake2[:, :, 1:, :] - fake2[:, :, :-1, :]).mean() +
                  torch.abs(fake2[:, :, :, 1:] - fake2[:, :, :, :-1]).mean())
        tv_loss += 0.5 * (torch.abs(fake1[:, :, 1:, :] - fake1[:, :, :-1, :]).mean() +
                  torch.abs(fake1[:, :, :, 1:] - fake1[:, :, :, :-1]).mean())

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        # identity loss 用来保证在油画转换为照片的时候保证颜色的偏差不会过大
        identity2 = gen_Z(photo2)
        identity1 = gen_H(photo1)
        identity_loss1 = l1(photo1, identity1)
        identity_loss2 = l1(photo2, identity2)

            # add all togethor
        G_loss = (
            loss_G_Z
            + loss_G_H
            + cycle_1_loss * config.LAMBDA_CYCLE_A
            + cycle_2_loss * config.LAMBDA_CYCLE_B  
            + identity_loss1 * config.LAMBDA_IDENTITY
            + identity_loss2 * config.LAMBDA_IDENTITY
            + tv_loss * config.LAMBDA_TV
        )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()



        if idx % 500 == 0:
            save_image(fake1, config.TRAINMODE+f"_train/Gen_picture_{idx/500}.png")
            save_image(fake2, config.TRAINMODE+f"_train/Gen_photo_{idx/500}.png")
        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))






def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = GANDataset(
        root1=config.TRAIN_DIR+config.TRAINMODE+"/trainA", 
        root2=config.TRAIN_DIR+config.TRAINMODE+"/trainB", 
        transform=config.transforms
    )
    val_dataset = GANDataset(
       root1=config.TRAIN_DIR+config.TRAINMODE+"/testA", 
       root2=config.TRAIN_DIR+config.TRAINMODE+"/testB", 
       transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )


    Mark = 1
    for epoch in range(config.NUM_EPOCHS):
        print(f"epoch:{epoch+1}:")

        #设置学习率衰减

        if (epoch>=100 and Mark==1):

            schedular_disc=torch.optim.lr_scheduler.StepLR(opt_disc,step_size=config.LR_DECAY,gamma=0.1);
            schedular_gen=torch.optim.lr_scheduler.StepLR(opt_gen,step_size=config.LR_DECAY,gamma=0.1);
            Mark=0

        
        train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse)

        if(epoch>=100):
            schedular_disc.step()
            schedular_gen.step()

    if config.SAVE_MODEL:
        save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
        save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
        save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
        save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()