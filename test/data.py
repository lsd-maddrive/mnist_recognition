
import numpy as np


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

low1=(100,185,0)
high1=(255,255,255)
limits1=[low1,high1]


res_img1=np.copy(test_img1)
mask1=np.zeros(test_img1.shape,dtype='uint8') 
mask1[0,1:]=1
res_img1=mask1*res_img1

test_img2=np.array([[[177,   6, 189],
        [ 51,  64, 168]],

       [[ 97,  71, 167],
        [212,  42,  83]]], dtype='uint8')

low2=(0,165,0)
high2=(170,255,255)
limits2=[low2,high2]

res_img2=np.copy(test_img2)
mask2=np.zeros(test_img2.shape,dtype='uint8') 
mask2[0,:,:]=1
res_img2=mask2*res_img2
