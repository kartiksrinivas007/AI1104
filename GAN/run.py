import os
import numpy as np

epoch_list = [2]
lr_list = np.logspace(-7, -1, num=7)

for epoch in epoch_list:
    for lr in lr_list:
        os.system(f"python GAN.py --epochs {epoch} --lr {lr}")
        pass
