# PhotoroomAssignment

Hi, thank you for reading and for the assignment; it was enjoyable. When the use of an image-to-image approach was mentioned, I immediately thought of a U-Net, but I couldn't recall the last time I implemented one, so I had to look up the typical architecture.

Then, I made the classic mistake of having the image data in the wrong format for PyTorch, which left me a bit puzzled, but I eventually resolved it.

Subsequently, I used MSE loss for optimization, which turned out to be a terrible idea, even though it only took a few minutes (optimizing on my own GPU may have been a mistake). The results were disappointing, leading to a bit of panic.

Afterward, I remembered that cross-entropy (KL divergence) measures the distance (I'm aware that KL is not symmetric) between distributions, so I thought that might be a better approach. Indeed, using that, I achieved results that look like this:

![Screenshot from 2023-12-05 18-38-37](https://github.com/szat/PhotoroomAssignment/assets/5555551/37948acd-71ac-4d4b-8bd9-d152325994e5)


And then after normalizing them, they look like this:


![Screenshot from 2023-12-05 18-52-23](https://github.com/szat/PhotoroomAssignment/assets/5555551/cbda79d4-6f4e-48af-9460-784eae9731ad)


Which makes me think it was the right path. Following would be to use contours from opencv to detect circles, and then the centers are the keypoints (so that is deterministic). 

# Libraries used:

- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import cv2
- import torch
- import torch.nn as nn
- import torch.nn.functional as F
- import pytorch_lightning as pl
- from torch.utils.data import Dataset, DataLoader, random_split

I apologize, there is a requirements.txt but I made the mistake of adding open3d just in case, before I started the assignement and it made the list really explode. Now I corrected that after the end of the assignment, I hope you don't mind since it has not much to do with the thought process. 

# Notes (after deadline):
- Two things that are really missing from my code is first data augmentation, then validation, and related to that the fact that my keypoint output is not ordered, so I need to find a solution for that. 
- I am finishing the problem for myself in the file after_time_limit.py, no need to look at it, I just dislike leaving things half done :)
