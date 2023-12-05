import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure I have access to my gpu
cuda_available = torch.cuda.is_available()
print(cuda_available)
# Ok great

df_train = pd.read_csv('training.csv')
print(f"DataFrame Shape {df_train.shape}")
df_train.head(2)

feature_col = 'Image'
target_cols = list(df_train.drop('Image', axis=1).columns)

# Fill missing values
df_train[target_cols] = df_train[target_cols].fillna(df_train[target_cols].mean())

# Image characteristics
IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_CHANNELS = 1

raw_images = np.array(df_train[feature_col].str.split().tolist(), dtype='float')
images = raw_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

labels = df_train[target_cols].values

def show_examples(images, landmarks):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for img, marks, ax in zip(images, landmarks, axes.ravel()):
        # Keypoints
        x_points = marks[:: 2]
        y_points = marks[1::2]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.scatter(x_points, y_points, s=10, color='red')
    plt.show()

# idx = np.random.choice(16, 16)
# show_examples(images[idx], labels[idx])

# Prepare the data
normalized_images = images.astype(np.float32) / 255.0
for i in range(normalized_images.shape[0]):
    img = normalized_images[i]
    mean = img.mean()
    std = img.std()
    img_new = (img - mean) / std
    normalized_images[i] = img_new

label_images = np.zeros_like(normalized_images)
labels_x = labels[:, 0::2]
labels_y = labels[:, 1::2]
# This is not very nice, but idk how to fix it atm, maybe with some heat map diffusion around subpixel center
labels_x_int = labels_x.astype(int)
labels_y_int = labels_y.astype(int)

# Sanity check
# tmp = normalized_images[0]
# tmp[labels_y_int[0], labels_x_int[0]] = 1
# Ok this looks fine

for i in range(len(label_images)):
    idx_y = labels_y_int[i]
    idx_x = labels_x_int[i]
    label_images[i, idx_y, idx_x] = 1

# Sanity check
# tmp = normalized_images[10]
# tmp2 = label_images[10]
# tmp[tmp2 == 1] = 1
# Ok this looks fine

# dims are not right, torch likes (N, C, H, W) = (N, 1, H, W)
# no patience to do it elegantly now
label_images = np.squeeze(label_images)
label_images = label_images[:, None, :, :]
print(label_images.shape)
normalized_images = np.squeeze(normalized_images)
normalized_images = normalized_images[:, None, :, :]
print(normalized_images.shape)

# I was having the classical problem that my channels were not in the right place, for a second I thought that
# the images had to be in the powers of 2, but actually not necessary, but some small code here to change the size:

# # Resize to be a multiple of 2, lets say to 128 to not lose information
# import scipy.ndimage
# zoom_f = 128 / 96
# zoom_f = (zoom_f, zoom_f, 1)
# # Resize the image using cubic interpolation
# label_images2 = np.zeros([len(label_images), 128, 128, 1])
# normalized_images2 = np.zeros([len(normalized_images), 128, 128, 1])
# #Takes way too long I think scipy is using CPU
# for i in range(len(label_images2)):
#     label_images2[i] = scipy.ndimage.zoom(label_images[i], zoom_f, order=3)
#
# for i in range(len(normalized_images2)):
#     normalized_images2[i] = scipy.ndimage.zoom(normalized_images[i], zoom_f, order=3)

from torch.utils.data import Dataset, DataLoader
class FaceDataset(Dataset):
    def __init__(self, images_np, labels_np):
        self.data = images_np
        self.label = labels_np

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.label[idx]).float()
        return image, label

from torch.utils.data import DataLoader, random_split
dataset = FaceDataset(normalized_images, label_images)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Lets code a basic u-net architecture for this, just something basic that will train
# I quickly googled github for some basic architecture and used it to remember how to do this:
# I know this is in tf not in torch
# https://github.com/zhixuhao/unet
# https://github.com/fel-thomas/simple_unet/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

pl.seed_everything(12)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(pl.LightningModule):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetBlock(1, 64)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.up1 = UNetBlock(512 + 256, 256)
        self.up2 = UNetBlock(256 + 128, 128)
        self.up3 = UNetBlock(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))

        # Decoder
        x_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.up1(x_up)
        x_up = F.interpolate(x_up, scale_factor=2, mode='bilinear', align_corners=True)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up2(x_up)
        x_up = F.interpolate(x_up, scale_factor=2, mode='bilinear', align_corners=True)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up3(x_up)

        out = self.out_conv(x_up)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.mse_loss(outputs, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

unet_model = UNet()
trainer = pl.Trainer(max_steps=1000, max_epochs=10)

torch.set_float32_matmul_precision('medium')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
trainer.fit(unet_model, train_loader, val_loader)

