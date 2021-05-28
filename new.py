import random, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision.all import *
from PIL import Image

path=Path('/home/ahmed_ragab/Downloads/archive/chest_xray')
#print(path.ls())
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

dls = data.dataloaders(path)
dls.valid.show_batch(max_n=12, nrows=2)
learn = cnn_learner(dls, resnet18, metrics=[error_rate,accuracy])
modell=learn.load('model_1.h5')
p=modell.predict('/home/ahmed_ragab/Downloads/archive/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
print(p[2])
#learn.fine_tune(3)
#learn.save('model_1.h5')
