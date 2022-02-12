
import cv2
import numpy as np
import pytest


@pytest.fixture
def test_data_1():
  test_img1=np.array(
     [[[ 96, 199, 129],
       [ 40,  30, 253],
       [ 26,   3, 168]],

     [[164,  59, 105],
       [244,  95, 252],
       [ 69, 152, 235]],

     [[100, 227,  30],
      [226, 108, 157],
      [ 54,   7,   7,]]]
     ,dtype='uint8')
  return test_img1


@pytest.fixture
def test_masks_1(test_data_1):
    low1=(100,185,0)
    high1=(255,255,255)
    mask_1=cv2.inRange(cv2.cvtColor(test_data_1, cv2.COLOR_RGB2HSV),low1,high1)
    list_masks_1=[mask_1]
    return list_masks_1


@pytest.fixture
def expected_result_1(test_data_1):
    res_img1=np.copy(test_data_1)
    mask1=np.zeros(test_data_1.shape,dtype='uint8') 
    mask1[0,1:]=1
    res_img1=mask1*res_img1
    return res_img1


@pytest.fixture
def test_data_2():
  test_img2=np.array([[[177,   6, 189],
        [ 51,  64, 168]],

       [[ 97,  71, 167],
        [212,  42,  83]]], dtype='uint8')
  return test_img2 


@pytest.fixture
def test_masks_2(test_data_2):
  low2=(0,165,0)
  high2=(170,255,255)
  mask_2=cv2.inRange(cv2.cvtColor(test_data_2, cv2.COLOR_RGB2HSV),low2,high2)
  list_masks_2=[mask_2]
  return list_masks_2

@pytest.fixture
def expected_result_2(test_data_2):
  res_img2=np.copy(test_data_2)
  mask2=np.zeros(test_data_2.shape,dtype='uint8') 
  mask2[0,:,:]=1
  res_img2=mask2*res_img2
  return res_img2
