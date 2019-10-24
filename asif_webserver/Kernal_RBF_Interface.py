# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:21:32 2016

@author: Asif Ahmad
"""


import numpy as np
from Kernalized_filter_Gaussian import Classifier
from Kernalized_filter_Gaussian import avgGaussianKernel

from skimage.io import imread, imshow
from skimage.color import rgb2gray , rgb2hed
from skimage import transform as tf
from skimage.measure import label

from scipy.misc import imsave
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter    #for blurring target that are have sharp transitions

from pylab import ginput,plot,show

import matplotlib.pyplot as plt

import timeit

import scipy.misc
import pylab


def apply(ifile,tfolder):
"""
Apply the filter using the trainingi mages placed in tfolder. The complete path of the input image is given in ifile.
"""
	################## Setting Paths #####################################################
	Images_Path = tfolder#'F:/PIEAS MS/4th Semester/THesis Research/Python_Coding/Kernalized_Filter_MOSSE/Kernel Large data set/Detection'        #All Images are stored in folder nammed "Detectin"
	########################################################################################

	I = rgb2gray(imread(Images_Path + '/img5/img5.bmp')) #Amina: Use os.path.join


	sI = np.shape(I)                              # Cell immages and targets both have same size .....


	sigma = 1                                    #for kernal,,
	lamda = 0.1                                  #regularization parameter
	alpha = 0.25                                 #percent width of tukey window
	sigma2 = 2.0                                 #sigma of gaussian filter used for blurring the target images


	#Rotations
	num_Rotations = 1                     
	######################################################################################

	examples = 100              #100 examples images
	###################################################################################

	##################Test Image#########################################################	
	test = imread(ifile)                               
	###########################################################################


	############# Training and Testing###############################################################################################	
	kfun = avgGaussianKernel
	Alpha = Classifier(size = sI, kfun = kfun, kparam = sigma, reg = lamda , sigma2 = sigma2, alpha = alpha)           #creating object of class classifier,, initialization     

	Alpha.Process_test_image (test)    

	for i in range(1,examples+1):       #All 99 images used in training except one image      
		 img=imread(Images_Path + '/img' + str(i) +'/img' + str(i) +'.bmp')                            #Oringinal Images #Amina: Use os.path.join
		 C = loadmat(Images_Path + '/img' + str(i) + '/img' + str(i) +'_detection.mat')['detection'] #True Locations of all cells	#Amina: Use os.path.join
		 C = np.array(C,np.int)
		 target = np.zeros(img.shape[:2])
		 for r,c in C:
			 target[c-1 , r-1]=1                                  #Desired Targets							
		 
		 Angle_Step = 360 / num_Rotations            #make it sure that angle step is integer value
		 for angle in range (0, 360, Angle_Step):
				img_rotated = tf.rotate (img , angle)
				target_rotated = tf.rotate (target , angle)
				if (np.max(target_rotated) != 0):
					 Alpha.ApplyCorrelationFilter(img_rotated ,target_rotated)      #Applying filter 
										  
	#################Actual Output of classifier###################################
	TestResponse = Alpha.TestResponse ()
	return TestResponse
	###########################################################################

if __name__=='__main__':
	img = apply(ifile,tfolder)
	
