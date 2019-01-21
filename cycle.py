import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image

from datasets import *
from models import Discriminator
from generator import Generator
from utils import  *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="horse2zebra", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr_g', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lr_d', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--sample_num', type=int, default=5)

opt = parser.parse_args()
print(opt)


# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)


# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_similar = nn.L1Loss()


# Initialize generator and discriminator
# G = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G = Generator(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

# if GPU is accessible
cuda = True if torch.cuda.is_available() else False
if cuda:
    G = G.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    criterion_similar.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Buffers of previously generated samples
# Buffers of previously generated samplesG
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
trans = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, transforms_=trans, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, transforms_=trans, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)

import numpy
path_horse = r'./data/horse2zebra/horse_example.jpg'
img_horse = Image.open(path_horse).convert('RGB')
image_horse = torch.Tensor(numpy.array(img_horse)).unsqueeze(0)
example_A = image_horse.permute(0, 3, 1, 2).cuda()
path_zebra = r'./data/horse2zebra/zebra_example.jpg'
img_zebra = Image.open(path_zebra).convert('RGB')
image_zebra = torch.Tensor(numpy.array(img_zebra)).unsqueeze(0)
example_B = image_zebra.permute(0, 3, 1, 2).cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))

    # Set model input
    layer_one = torch.ones(5, 1, opt.img_width, opt.img_height)
    layer_zero = torch.zeros(5, 1, opt.img_width, opt.img_height)
    label_A = torch.cat((layer_one, layer_zero), 1).cuda()
    label_B = torch.cat((layer_zero, layer_one), 1).cuda()

    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))

    real_A_ = Variable(torch.cat((real_A, label_A), 1))
    real_B_ = Variable(torch.cat((real_B, label_B), 1))

    examples_A = example_A
    examples_B = example_B
    for i in range(4):
        examples_A = torch.cat((examples_A, example_A), 0)
        examples_B = torch.cat((examples_B, example_B), 0)
    image_A_ = Variable(torch.cat((examples_A, label_A), 1))
    image_B_ = Variable(torch.cat((examples_B, label_B), 1))

    fake_B = G(real_A_, image_B_)
    fake_A = G(real_B_, image_A_)

    img_sample = torch.cat((real_A.data, fake_B.data,
                            real_B.data, fake_A.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done),
                nrow=5, normalize=True)
    '''
        for i in range(5):
            imgs = next(iter(val_dataloader))

            layer_one = torch.ones(opt.batch_size, 1,opt.img_width, opt.img_height)
            layer_zero = torch.zeros(opt.batch_size, 1,opt.img_width, opt.img_height)
            label_A = torch.cat((layer_one,layer_zero), 1).cuda()
            label_B = torch.cat((layer_zero,layer_one), 1).cuda()

            real_A = imgs['A'].type(Tensor)
            img_A = torch.cat((real_A, label_A),1)
            real_A_ = Variable(img_A)

            real_B = imgs['B'].type(Tensor)
            img_B = torch.cat((real_B, label_B),1)
            real_B_ = Variable(img_B)

            # fake_A, fake_B = G(real_A_, real_B_)
            fake_B, fake_A = G(real_A, real_B)
            if i == 0:
                img_sample = torch.cat((real_A.data, fake_B.data,
                                    real_B.data, fake_A.data), 0)
            else:
                img = torch.cat((real_A.data, fake_B.data,
                                        real_B.data, fake_A.data), 0)
                img_sample = torch.cat((img_sample, img), 0)
        save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done),
                   nrow=4, normalize=True)
     '''



# ----------
#  Training
# ----------
prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    # Loss weights
    lambda_GAN = 3
    lambda_cyc = 7
    lambda_id = 0.5 * lambda_cyc

    for i, batch in enumerate(dataloader):

        # Set model input
        layer_one = torch.ones(opt.batch_size, 1,opt.img_width, opt.img_height)
        layer_zero = torch.zeros(opt.batch_size, 1,opt.img_width, opt.img_height)
        label_A = torch.cat((layer_one,layer_zero), 1).cuda()
        label_B = torch.cat((layer_zero,layer_one), 1).cuda()

        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        real_A_ = Variable(torch.cat((real_A, label_A), 1))
        real_B_ = Variable(torch.cat((real_B, label_B), 1))
        image_A_ = Variable(torch.cat((example_A, label_A), 1))
        image_B_ = Variable(torch.cat((example_B, label_B), 1))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), 1, opt.img_height // 2**4, opt.img_height // 2**4))),
                         requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), 1, opt.img_height // 2**4, opt.img_height // 2**4))),
                        requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        fake_B = G(real_A_, image_B_)
        fake_A = G(real_B_, image_A_)
        recov_A = G(torch.cat((fake_B, label_B),1), image_A_)
        recov_B = G(torch.cat((fake_A, label_A),1), image_B_)


        # Identity loss
        loss_id_B = criterion_identity(G(real_B_, image_B_), real_B)
        # GAN loss
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        # Cycle loss
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        # Total loss
        loss_G_AB = loss_GAN_AB + lambda_cyc * loss_cycle_B + lambda_id * loss_id_B #+ lambda_sml * loss_similar_B

        # Identity loss
        loss_id_A = criterion_identity(G(real_A_, image_A_), real_A)
        # GAN loss
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        # Cycle loss
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        # Total loss
        loss_G_BA = loss_GAN_BA + lambda_cyc * loss_cycle_A + lambda_id * loss_id_A # + lambda_sml * loss_similar_A

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        loss_identity = (loss_id_A + loss_id_B) / 2
        loss_G = (loss_G_AB + loss_G_BA) / 2

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] "
                         "[D loss: %.4f] "
                         "[G loss: %.4f, adv: %.4f, cycle: %.4f, identity: %.4f] "
                         "ETA: %s" %
                         (epoch+1, opt.n_epochs,
                          i+1, len(dataloader),
                          loss_D.item(),
                          loss_G.item(), loss_GAN.item(), loss_cycle.item(),
                          loss_identity.item(), time_left))
        #, loss_similar: %f loss_similar.item(),

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), 'saved_models/%s/G_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))
