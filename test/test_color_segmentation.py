import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(PROJECT_ROOT)

from object_detection.segmentation.color_segmentation import color_segmentation
from test.data import *

import data
import numpy as np



def test_1():
    assert np.all(color_segmentation(data.test_img1,data.limits1) == data.res_img1)



def test_2():
    assert np.all(color_segmentation(data.test_img2,data.limits2) == data.res_img2)

