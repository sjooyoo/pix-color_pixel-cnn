import os
import numpy as np
import time
import datetime
import torch
import skimage.color as sc


class Timer():
    def __init__(self):
        self.cur_t = time.time()

    def tic(self):
        self.cur_t = time.time()

    def toc(self):
        return time.time() - self.cur_t

    def tocStr(self, t=-1):
        if (t == -1):
            return str(datetime.timedelta(seconds=np.round(time.time() - self.cur_t, 3)))[:-4]
        else:
            return str(datetime.timedelta(seconds=np.round(t, 3)))[:-4]


def make_folder(path, dataset):
    try:
        os.makedirs(os.path.join(path, dataset))
    except OSError:
        pass


def data_process(image_data, batch_size, imsize):

    images_numpy = image_data.numpy()
    input = torch.zeros(batch_size, 1, imsize, imsize)
    labels = torch.zeros(batch_size, 2, imsize, imsize)
    for k in range(batch_size):
        rgb = images_numpy[k].transpose(1, 2, 0)
        yCbCr = sc.rgb2ycbcr(rgb) / 255
        img_y = yCbCr[:, :, 0]
        input[k] = torch.from_numpy(np.expand_dims(img_y,0))
        img_CbCr = yCbCr[:, :, 1:3]
        labels[k] = torch.from_numpy(img_CbCr)

    return input, labels







