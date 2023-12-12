# PhotoroomAssignment

Hi, thank you for reading and for the assignment; it was enjoyable. When the use of an image-to-image approach was mentioned, I immediately thought of a U-Net, but I couldn't recall the last time I implemented one, so I had to look up the typical architecture.

Then, I made the classic mistake of having the image data in the wrong format for PyTorch, which left me a bit puzzled, but I eventually resolved it.

Subsequently, I used MSE loss for optimization, which turned out to be a terrible idea, even though it only took a few minutes (optimizing on my own GPU may have been a mistake). The results were disappointing, leading to a bit of panic.

Afterward, I remembered that cross-entropy (KL divergence) measures the distance (I'm aware that KL is not symmetric) between distributions, so I thought that might be a better approach. On the left is the output, on the right after I normalize them. 

<p>
  <img src="https://github.com/szat/PhotoroomAssignment/assets/5555551/37948acd-71ac-4d4b-8bd9-d152325994e5"
 width="300" height="300" />
  <img src="https://github.com/szat/PhotoroomAssignment/assets/5555551/cbda79d4-6f4e-48af-9460-784eae9731ad"
 width="300" height="300" />

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
- import sklearn

After the deadline I fixed the requirement.txt so that the vm can be installed without problems. <del> I apologize, there is a requirements.txt but I made the mistake of adding open3d just in case, before I started the assignement and it made the list really explode. Now I corrected that after the end of the assignment, I hope you don't mind since it has not much to do with the thought process. </del>

# Notes (after deadline):
- Two things that are really missing from my code is first data augmentation, then validation, and related to that the fact that my keypoint output is not ordered, so I need to find a solution for that. 
- I am playing with the problem for myself in the file after_time_limit.py, no need to look at it :)
- Visual inspection of the case of 3 distant keypoints encoded into the same channel and separated via mixture of gaussians. 

![3pts_2eyes_1mouth](https://github.com/szat/PhotoroomAssignment/assets/5555551/dfba58a8-3bc6-45e0-9f17-4616dbeea36c)

- Visual inspection of the case of 6 points that are closer to each other, 3 for each eyes, using mixture of gaussians again. 

![6pts_3eyes_3eyes](https://github.com/szat/PhotoroomAssignment/assets/5555551/be1f352e-b01a-4ad9-a306-da696a0b0c8c)

- Visual inspection of the case of all 15 points but from 1 channel only, not one-hot encoding, using a mixture of gaussians again.
  
![15pts_1channel](https://github.com/szat/PhotoroomAssignment/assets/5555551/0d64b58e-d89b-4da9-9256-8104be860254)
