import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import futils

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='./flowers/test/20/image_04910.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
path = pa.checkpoint
cat_names=pa.category_names

training_loader, testing_loader, validation_loader, train_data = futils.load_data()

model = futils.load_checkpoint(path)

if cat_names:
    with open(cat_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    prob, classes = futils.predict(path_image, model, number_of_outputs, power)
    print('File selected: ' + path_image)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])

