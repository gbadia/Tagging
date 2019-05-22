# -*- coding: utf-8 -*-
import json
from skimage import io
from skimage.transform import rescale
import numpy as np
import ColorNaming as cn

import os.path
import Labels as lb
import KMeans as km


Options={'colorspace':'RGB', 'K':6, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False, 'max_iter': 1}


im = io.imread('Images/0010.png')
k_m = km.KMeans(im, Options['K'], Options)
k_m.run()
fisher = k_m.fitting()

print(fisher)
