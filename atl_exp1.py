import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.mixture import GaussianMixture

FOLDER = "/home/adrian/Code/PhotoroomAssignment/"

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
# Different strategies:
# - [0,1] normalization
# - normalization across dataset, feature-wise centering
# - normalization across samples, sample-wise centering
# https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/

normalized_images = images.astype(np.float32) / 255.0
for i in range(normalized_images.shape[0]):
    img = normalized_images[i]
    mean = img.mean()
    std = img.std()
    img_new = (img - mean) / std
    normalized_images[i] = img_new

# To get precise kpts locations, no need to change input images, just put the points on upscaled canvas.
# Then we can either downsize or train the network on that output size

labels_x = labels[:, 0::2]
labels_y = labels[:, 1::2]
# This is not very nice, but idk how to fix it atm, maybe with some heat map diffusion around subpixel center
# Unsure weather we should have points added 0.5 or not...

# Check whether it should be adjusted by 0.5 or not:
# def show_examples_adjusted(images, landmarks, adjust=False):
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
#     for img, marks, ax in zip(images, landmarks, axes.ravel()):
#         # Keypoints
#         x_points = marks[:: 2]
#         y_points = marks[1::2]
#         if adjust:
#             x_points += 0.5
#             y_points += 0.5
#         ax.imshow(img.squeeze(), cmap='gray')
#         ax.scatter(x_points, y_points, s=2,marker='x',color='red')
#     plt.show()

# idx = np.random.choice(16, 16)
# show_examples_adjusted(images[idx], labels[idx], False)
# Upon visual inspection, we should not correct with 0.5

labels_x_int = labels_x.astype(int)
labels_y_int = labels_y.astype(int)

# Sanity check
# tmp = np.squeeze(copy.deepcopy(normalized_images[0]))
# dimensions = (IMG_WIDTH*4, IMG_HEIGHT*4)
# tmp2 = cv2.resize(tmp, dimensions, interpolation=cv2.INTER_AREA)
# tmp2[labels_y_int[0], labels_x_int[0]] = 1
# Ok this looks fine

# Sanity check with cv2.circle (row/col vs x/y)
# i = 0
# tmp = np.zeros([96, 96])
# for j in range(15):
#     idx_y = labels_y_int[i][j]
#     idx_x = labels_x_int[i][j]
#     tmp = cv2.circle(tmp, [idx_x, idx_y], 2, 255, -1)
# This is ok

label_images = np.zeros([len(normalized_images), 1, 96, 96])
for i in range(len(label_images)):
    tmp = np.zeros([96, 96])
    j = 0
    idx_y = labels_y_int[i][j]
    idx_x = labels_x_int[i][j]
    tmp = cv2.circle(tmp, [idx_x, idx_y], 2, 255, -1)

    j = 5
    idx_y = labels_y_int[i][j]
    idx_x = labels_x_int[i][j]
    tmp = cv2.circle(tmp, [idx_x, idx_y], 2, 255, -1)

    j = 10
    idx_y = labels_y_int[i][j]
    idx_x = labels_x_int[i][j]
    tmp = cv2.circle(tmp, [idx_x, idx_y], 2, 255, -1)

    tmp /= 255.0
    tmp = cv2.GaussianBlur(tmp, (3, 3), 1)
    label_images[i] = tmp

tmp4 = np.zeros([96, 96])
for j in range(15):
    i = 0
    idx_y = labels_y_int[i][j]
    idx_x = labels_x_int[i][j]
    tmp4 = cv2.circle(tmp4, [idx_x, idx_y], 2, 255, -1)
tmp4 /= 255.0
tmp4 = cv2.GaussianBlur(tmp4, (3, 3), 1)


tmp = label_images[0, 0]
tmp2 = np.squeeze(normalized_images[0])

tmp3 = np.zeros([96,96,3])
tmp3[:,:,0] = tmp
tmp3[:,:,1] = tmp4
tmp3[:,:,2] = tmp2

normalized_images = np.squeeze(normalized_images)
normalized_images = normalized_images[:, None, :, :]
print(normalized_images.shape)
print(label_images.shape)

class FaceDataset(Dataset):
    def __init__(self, images_np, labels_np, transform=None):
        self.data = images_np
        self.label = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.label[idx]).float()
        return image, label

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
    def __init__(self, error_threshold=0.001):
        super(UNet, self).__init__()
        self.error_threshold = error_threshold

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
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_validation_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None and val_loss < self.error_threshold:
            print(f"stop training, val loss = {val_loss:.4f}, thresh = {self.error_threshold:.4f}")
            self.trainer.should_stop = True

unet_model = UNet()
trainer = pl.Trainer(max_steps=3000, max_epochs=15)

torch.set_float32_matmul_precision('medium')
trainer.fit(unet_model, train_loader, val_loader)
#
# trainer.validate(unet_model, val_loader)
# checkpoint = {
#     'state_dict': unet_model.state_dict(),
#     'optimizer': trainer.optimizers[0].state_dict(),
#     # Include any other components you need to save
# }
#
# torch.save(checkpoint, FOLDER + 'model_checkpoint.ckpt')
# checkpoint = torch.load(FOLDER+'model_checkpoint.ckpt')
#
# # Initialize your model
# model = UNet()
#
# # Apply the saved state
# model.load_state_dict(checkpoint['state_dict'])


# Lets look at output if it makes sense
# Set model to eval mode, disable gradients
unet_model.eval()
img_list = []
kpts_pred_list = []
kpts_true_list = []
val_data_iter = iter(val_dataset)

for i in range(8):
    test_img, test_label = next(val_data_iter)
    test_img = test_img[None, :, :, :]
    with torch.no_grad():
        output = unet_model(test_img)
    test_output_np = np.squeeze(np.array(output))
    test_output_np[test_output_np < 0] = 0
    test_output_np /= test_output_np.max()

    # mean = tmp.mean()
    # std = tmp.std()
    # tmp = (tmp - mean) / std
    # tmp /= tmp.max()
    # tmp[tmp < 0] = 0

    img_list.append(test_output_np)
    tmp = test_output_np

    test_img = np.squeeze(np.array(test_img))
    test_label = np.squeeze(np.array(test_label))

    # Convert heatmap to a set of points
    x, y = np.indices(tmp.shape)
    x = x.ravel()
    y = y.ravel()
    values = tmp.ravel()

    # Create a realistic distribution of points
    scaled_values = (values * 100).astype(int)
    points = np.vstack([x, y]).T.repeat(scaled_values, axis=0)

    # Fit a Gaussian Mixture Model
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(points)

    ind_x = gmm.means_[:, 0]
    ind_y = gmm.means_[:, 1]

    ind_pred = []
    for k in range(len(ind_x)):
        ind_pred.append((ind_x[k], ind_y[k]))

    # The right way would be to just change the data loader, but im too lazy for that now
    test_label = np.squeeze(np.array(test_label))
    x, y = np.indices(test_label.shape)
    x = x.ravel()
    y = y.ravel()
    values = test_label.ravel()

    # Create a realistic distribution of points
    scaled_values = (values * 100).astype(int)
    points = np.vstack([x, y]).T.repeat(scaled_values, axis=0)

    # Fit a Gaussian Mixture Model
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(points)

    ind_x = gmm.means_[:, 0]
    ind_y = gmm.means_[:, 1]

    ind_true = []
    for k in range(len(ind_x)):
        ind_true.append((ind_x[k], ind_y[k]))

    kpts_pred_list.append(ind_pred)
    kpts_true_list.append(ind_true)

# Print the parameters of the Gaussians
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)

# Optionally, visualize the results
plt.imshow(tmp, cmap='hot', interpolation='nearest')
plt.scatter(gmm.means_[:, 1], gmm.means_[:, 0], c='blue', marker='x')
plt.title("Heatmap with GMM centers")
plt.show()

img_list = []
kpts_pred_list = []
kpts_true_list = []

for i in range(8):
    test_img, test_label = next(val_data_iter)

    test_img = test_img[None, :, :, :]
    with torch.no_grad():
        output = unet_model(test_img)
    test_output_np = np.squeeze(np.array(output))
    test_output_np[test_output_np < 0] = 0
    tmp = test_output_np
    tmp /= tmp.max()
    # mean = tmp.mean()
    # std = tmp.std()
    # tmp = (tmp - mean) / std
    # tmp /= tmp.max()
    # tmp[tmp < 0] = 0

    img_list.append(tmp)

    # ind_pred = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
    ind_true = np.unravel_index(np.argmax(test_label, axis=None), tmp.shape)

    # pred_list = []
    true_list = []

    # pred_list.append(ind_pred)
    true_list.append(ind_true)

    # kpts_pred_list.append(pred_list)
    kpts_true_list.append(true_list)

# We assume we have max 15 kpts
rainbow_colors = ["red", "orange", "yellow", "green", "blue",
                  "indigo", "violet", "red", "orange", "yellow",
                  "green", "blue", "indigo", "violet", "red",
                  "orange", "yellow", "green", "blue", "indigo",
                  "violet", "red", "orange", "yellow", "green"]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
fig.suptitle("Marker x for predicitions, o for ground truth." , fontsize=14)
tmp_ax = axes.ravel()
for i in range(8):
    img = img_list[i]
    ax = tmp_ax[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for j, kpts in enumerate(kpts_pred_list[i]):
        ax.scatter(kpts[1], kpts[0], s=40, marker='x', color=rainbow_colors[0])
    for j, kpts in enumerate(kpts_true_list[i]):
        ax.scatter(kpts[1], kpts[0], s=40, marker='o', facecolors='none', color=rainbow_colors[0])
fig.tight_layout()
plt.show()


#
# plt.imshow(img_list[0], cmap='gray')
# plt.scatter(kpts_pred_list[0][1], kpts_pred_list[0][0], s=10, marker='x',color='red')
# plt.scatter(kpts_true_list[0][1], kpts_true_list[0][0], s=10, marker='o',color='red')
# plt.show()
#





#
# idx = np.random.choice(100, 8)
# tmp_images = images[idx]
# tmp_labels = labels[idx]
#
# show_predictions(images[idx])
#
#
# # plt.imshow(tmp, cmap='gray')  # Use cmap='gray' for grayscale images
# # plt.axis('off')
# # plt.show()
#
#
# # Ok this is def not good at all, it is a weird heatmap, but not want we want.
#
# # Hmm interesting discussion here:
# # https://datascience.stackexchange.com/questions/51048/mse-vs-cross-entropy-for-training-with-facial-landmark-pose-heatmaps
#
# # Ok this was definitely the right way to go, I dont have much more time to train want to try to finish the actual
# # Keypoint extraction, for which I'll try to use open-cv.
# # To be done better:
# # - data augmentation, flips, rotation, affine transforms (to simulate 3d pose changes)
# # - augment the number of layers: bias - var tradeoff actually does not seem to depend on the nb of parameters:
# # https://arxiv.org/abs/1810.08591
# # - maybe do not dilate or blurr the blobs, it seems that the heatmaps act too much like heatmaps and fuse together.
# # - train longer
#
# # Other thoughts: recall is which keypoints were detected like actual kpts (blobs), accuracy will tbe center of a blob
# # Maybe compare different models using F1 score, since both metrics seem important here (especially if we have faces
# # where you can see only part of a face, i.e. the general case, there we dont want the model to be forced to find 15,
# # so there recall seems more important).
#
# # So in case we have occlusions: prioritize recall, if it is always the full face, prioritize precision.
#
# # Lets try to find the blobs:
# # Lets just go with this: https://stackoverflow.com/questions/74771948/how-to-localize-red-regions-in-heatmap-using-opencv
#
# print(test_output_np.max())
# # Normalize again the output like we did for the input
# test_output_np /= test_output_np.max()
#
# mean = test_output_np.mean()
# std = test_output_np.std()
# test_output_np_new = (test_output_np - mean) / std
#
# mask = test_output_np_new > 2  # lets pick 2 standard dev
#
# # Then find contours and then find the center of the contours which will give deterministically the location of the keypoint.
#
# # Running out of time here. Thanks for taking the pain of reading up to here.
