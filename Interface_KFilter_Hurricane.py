#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:16:45 2017
@author: bismillah
"""


import numpy as np
from kerkor import Classifier #kerkor, this name was given by Dr. FAAM, :P
from kerkor import avgGaussianKernel
from skimage.io import imread, imshow
from skimage.color import rgb2gray , rgb2hed
from skimage import transform as tf
from skimage.measure import label
from scipy.misc import imsave
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter    #for blurring target that are have sharp transitions
from pylab import ginput,plot,show
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from joblib import Parallel, delayed
import scipy.misc
import pylab
import time
import glob
################## Setting Paths #####################################################
                 
savePath='./Simulation_results/' #path where results are going to store
train_path=glob.glob('./Combine_Data/*')
train_path.sort()

#All Thresholded Images are stored in folder named 'ThresholdedImages'      
###################################################################################
sI = (300,300)                            # Size of each hurricane
sigma = 3.0                               #for gaussian kernal
lamda = 0.5                               #regularization parameter
alpha = 0.25      #0.25 to 0.5            #percent width of tukey window
sigma2 = 2.0     #I'm not using this    #sigma of gaussian filter used for blurring the target images

#Creat target images, gaussian of sigma=5.0 in center of the 300x300 image
target=np.zeros(sI)
target[150,150]=1.0     
target=gaussian_filter(target, 5.0) 

errorK=0
inten=[] #contains intensity values for plotting 

#Divide data into 5-folds
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
allData=np.array(train_path)

test_data=[]
train_data=[]

#Separate testing and train data for each fold
for trainIdx, testIdx in kf.split(train_path):
    test_data.append(allData[testIdx]) 
    train_data.append(allData[trainIdx])  

fold=1 #for five folds hots will have value of 0, 1, 2, 3, 4. Here I am taking results for only one fold=1
testData=test_data[fold]
trainData=train_data[fold]
def myFunction(j):
    name=testData[j].split('/')    #split the name where slach occurs
    #stime=time.time()
    print 'Processing Folder ', name[-1]
    kfun = avgGaussianKernel
    Alpha = Classifier(size = sI, kfun = kfun, kparam = sigma, reg = lamda , sigma2 = sigma2, alpha = alpha)           
    testImgs=glob.glob(testData[j]+'/*') #retreive all test images path of each folder (hurricane)
    error=[]
    for m in range(len(testImgs)): #now read the test images one by one
        test = imread(testImgs[m]) #Reading test image
        intt=testImgs[m].split('.')
        intt=intt[9]                    #get the Intensity value of the hurricane from the name of image
        Alpha.Process_test_image (test)    #process test image, function is found in kerkor file
        #inten.append(int(intt))
        for i in range(len(trainData)):  
            trainImgs=glob.glob(trainData[i]+'/*') #Retreive train images
            for k in range(len(trainImgs)):  
                img=imread(trainImgs[k])
                Alpha.ApplyCorrelationFilter(img ,target) #add to the correlaton according to the mathematical
                                                          #formulation of the kernelized filter
                  
        """________Response of classifier_______"""
        #  etime=time.time()
        #  ttime=etime-stime
        #print 'Time taken: ', ttime
        TestResponse = Alpha.TestResponse () #get response for each test image
            
        #fig,ax = plt.subplots(1)    
        # Display the image
        ##ax.imshow(test)    
        # Create a Rectangle patch
        center=np.unravel_index(TestResponse.argmax(), TestResponse.shape) #get maximum value in the image and consider it as center
        #pos=(center[1], center[0])
        #maxPos=np.subtract(pos,(5,5))
        #rect = patches.Rectangle(maxPos,10,10,linewidth=1,edgecolor='w',facecolor='none')
        errorK=np.sqrt((150-center[0])**2+(150-center[1])**2) #calculate error between real center (150,150)
                                                              #and the center detected by kernelized filter       
        
        # Add the patch to the Axes
        #ax.add_patch(rect)
        #plt.title('Intensity '+intt)
        #fig.savefig(savePath+str(j)+"._"+intt+'.png')
        ##plt.show()
       
        #plt.figure()
        ##plt.imshow(TestResponse, cmap=plt.cm.jet)
        #plt.title(str(maxPos))
        #plt.imsave(savePath+str(j)+"._"+str(maxPos)+'.png', TestResponse, cmap=plt.cm.jet)
        error.append((m, int(intt), errorK))
    name=testData[j].split('/')    
    np.save(savePath+name[-1]+"-"+str(fold)+'-'+str(j)+".npy", error) #save result for each test folder
    #length of "error" array will be equal to the number of test images in folder j.
                                                            
    

if __name__=='__main__':
    Parallel(n_jobs=8)(delayed(myFunction)(j) for j in range(len(testData)))
        
    
    #np.save(savePath+"result"+str(host)+".npy", res)
    #Commented section is used for plotting the dual axis graph
    """
    tt=np.arange(len(testData)) 
    fig, ax1 = plt.subplots()
    #plot kernelized error
    ax1.plot(tt, res[2], 'g-', label=testData'Kernelized Filter')
    ax1.legend(loc='upper right')
    
    ax1.set_xlabel('Images')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Error')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(tt, res[1], 'r.')
    
    ax2.set_ylabel('Intensity (Knots)', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(savePath+"Error_Plot.png")
    """