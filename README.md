# PhotoroomAssignment

Hi thanks for reading and thanks for the assignment it was fun. When it was mentionned to use an image to image approach, I thought of a U-net right away, but I cold not even remember last time I implemented one, so I had to look up what is the usual architecture. 

Then  I made the classical mistake that the image data was not in the right order for pytorch, which left me stracthing my head a bit, but then that was solved. 

Following, I optimized used MSE loss, which was a terrible idea but took a few minutes (optimizing on my own GPU, maybe a mistake). The results were disappointing, and so panic happened. 

Following, I remembered that cross entropy (KL divergence) encodes distance (I know KL is not symmetric) between distributions, so maybe that would be better. Indeed with that I got results that looks like this:


![Screenshot from 2023-12-05 18-38-37](https://github.com/szat/PhotoroomAssignment/assets/5555551/37948acd-71ac-4d4b-8bd9-d152325994e5)


And then after normalizing them, they look like this:


![Screenshot from 2023-12-05 18-52-23](https://github.com/szat/PhotoroomAssignment/assets/5555551/cbda79d4-6f4e-48af-9460-784eae9731ad)


Which makes me think it was the right path. Following would be to use contours from opencv to detect circles, and then the centers are the keypoints (so that is deterministic). 

# Libraries used:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

I apologize, there is a requirements.txt but I made the mistake of adding open3d just in case, before I started the assignement and it made the list really explode. 
