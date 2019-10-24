# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:12:18 2016

@author: Asif Ahmad
"""

import numpy as np


from Linear_MOSSE_filter import MOSSE



from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hed
from skimage import transform as tf
from pylab import ginput,plot,show
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter 

import matplotlib.pyplot as plt

from scipy.io import loadmat

from skimage.measure import label

import timeit







     




################## Setting Paths #####################################################

Images_Path = 'F:/PIEAS MS/4th Semester/THesis Research/Python_Coding/Kernalized_Filter_MOSSE/Kernel Large data set/Detection'        #All Images are stored in folder nammed "Detectin"
###################################################################################


I = rgb2gray(imread(Images_Path + '/img5/img5.bmp'))


sI = np.shape(I)          # Cell immages and targets both have same size .....
s=5
lamda = 0.1           #regularization parameter

sigma2 = 2.0         #for blurring of desired target images
alpha = 0.25         #tukey

examples = 100



###################Test Image########################################################
j = 55
test_image = imread(Images_Path + '/img' + str(j) +'/img' + str(j) +'.bmp') 
test = rgb2hed(test_image)[:,:,0]          #hematoxyling features only
######################################################################################


##################Training################################################################
H=MOSSE(sI, lamda, sigma2, alpha)

for i in range(1,examples + 1):    #Training On all Images 99 except test image
     if (i != j):
         
         img=rgb2hed(imread(Images_Path + '/img' + str(i) +'/img' + str(i) +'.bmp')) 
         img = img[:,:,0]       #only hematoxylin features
         
         C = loadmat(Images_Path + '/img' + str(i) + '/img' + str(i) +'_detection.mat')['detection'] #all type of cells
    
         C = np.array(C,np.int)
         target = np.zeros(img.shape[:2])
         for r,c in C:
             target[c-1,r-1]=1                           #Desried Target Images

         if (np.max(target) != 0):
             H.addtraining(img,target)
##################################################################################33

##############################Filter##############################################
filterM = H.imagefilter()
##########################################################


##################Testing#######################################################        

#########################Actual Response of Filter################################
TestResponse = H.correlate(test)        #actural target of correlation

########################## Desired Response ###############################    
C = loadmat(Images_Path + '/img' + str(j) + '/img' + str(j) +'_detection.mat')['detection']

C = np.array(C,np.int)
t = np.zeros(test.shape[:2])
for r,c in C:
    t[c-1,r-1]=1    
t = gaussian_filter(t,sigma2)             

#######################################################################################
    


########################################Plots###########################################
plt.figure(1)
imshow(test_image)              #test image
plt.title('Test Image')

plt.figure(2)
imshow(TestResponse)
plt.title('Linear MOSSE Response')

plt.figure(3)
imshow(t)
plt.title('Desired Target Locations')

plt.figure(4)
imshow(filterM)
plt.title('Linear MOSSE Trained Filter')

plt.show()
#######################################################################################                      



