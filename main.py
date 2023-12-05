import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

idx = np.random.choice(16, 16)
show_examples(images[idx], labels[idx])

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
tmp = normalized_images[0]
tmp[labels_y_int[0], labels_x_int[0]] = 1
# Ok this looks fine

for i in range(len(label_images)):
    idx_y = labels_y_int[i]
    idx_x = labels_x_int[i]
    label_images[i, idx_y, idx_x] = 1

# Sanity check
tmp = normalized_images[10]
tmp2 = label_images[10]
tmp[tmp2 == 1] = 1
# Ok this looks fine

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

# Create the dataset
dataset = FaceDataset(normalized_images, label_images)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



