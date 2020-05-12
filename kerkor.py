# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:51:52 2016

@author: Asif Ahmad

"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 23:43:57 2016

@author: Asif Ahmad

RBF Correlation filter implementation along with preprocessing steps,
"""


import numpy as n
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray , rgb2hed
import matplotlib.pyplot as plt
from skimage import transform as tf
from scipy.misc import imsave
import scipy.misc
import pylab
from skimage.morphology import disk
from skimage.filters.rank import gradient
from scipy.ndimage.filters import gaussian_filter    #for blurring target that are have sharp transitions
from scipy.ndimage.filters import gaussian_laplace
from scipy.ndimage.filters import sobel

def avgGaussianKernel(sigma,x,y = None):
    k = n.zeros(x.shape[:2],n.complex128)   
    x = np.atleast_3d(x)                   
    for i in range(0 , x.shape[2]):
        if y is None:
            k += dense_gauss_kernel(sigma, x[:,:,i])  
        else:
            k += dense_gauss_kernel(sigma, x[:,:,i], y[:,:,i])            
    k = k / x.shape[2]
    return k
    
def preprocess(I):
    """
    Receives a raw image, it preprocesses and sets the channels
    Input: MxNxD Image
    Output: MxNxP Image    
    """
    I=I[:300,:300,:]
    Zg=rgb2gray(I)
    
    S=sobel(Zg, cval=0.0)
    LG=gaussian_laplace(Zg, 3.0)
    ZZ = np.array((Zg,S, LG)).T
    #print ZZ
    #ZZ = np.array((Zg,V)).T
    ZZ=meanunit(ZZ)  
    return ZZ
        
def dense_gauss_kernel(sigma, x, y=None):
    """
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must alsoc
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    """

    xf = pylab.fft2(x)  # x in Fourier domain
    x_flat = x.flatten()
    xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

    if y is not None:
        # general case, x and y are different
        yf = pylab.fft2(y)
        y_flat = y.flatten()
        yy = pylab.dot(y_flat.transpose(), y_flat)
    else:
        # auto-correlation of x, avoid repeating a few operations
        yf = xf
        yy = xx

    # cross-correlation term in Fourier domain
    xyf = pylab.multiply(xf, pylab.conj(yf))

    # to spatial domain
    xyf_ifft = pylab.ifft2(xyf)

    xy = pylab.real(xyf_ifft)
    # calculate gaussian response for all positions
    scaling = -1 / (sigma**2)
    xx_yy = xx + yy
    xx_yy_2xy = xx_yy - 2 * xy
    k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy))

    #print("dense_gauss_kernel x.shape ==", x.shape)
    #print("dense_gauss_kernel k.shape ==", k.shape)

    return k




def tukeywindow(size, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 '''
    # Special cases
#    if alpha <= 0:
#        return np.ones(window_length) #rectangular window
#    elif alpha >= 1:
#        return np.hanning(window_length)
    w,h = size
    
    # Normal case
    #Window in x direction (vertical direction)
    X = n.arange(w).reshape(w,1)
    X = X / (w-1.0)
    X = X*n.ones((1,h),'d')
    window_X = n.ones(X.shape) 
    # first condition 0 <= x < alpha/2
    first_condition = X<alpha/2
    window_X[first_condition] = 0.5 * (1 + n.cos(2*n.pi/alpha * (X[first_condition] - alpha/2) )) 
    # second condition already taken care of 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X>=(1 - alpha/2)
    window_X[third_condition] = 0.5 * (1 + n.cos(2*n.pi/alpha * (X[third_condition] - 1 + alpha/2)))     
        
    #Window in horizontal directions 
    Y = n.arange(h).reshape(1,h)
    Y = Y / (h-1.0)
    Y = Y*n.ones((w,1),'d')
    window_Y = n.ones(Y.shape) 
    # first condition 0 <= x < alpha/2
    first_condition = Y<alpha/2
    window_Y[first_condition] = 0.5 * (1 + n.cos(2*n.pi/alpha * (Y[first_condition] - alpha/2) )) 
    # second condition already taken care of 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y>=(1 - alpha/2)
    window_Y[third_condition] = 0.5 * (1 + n.cos(2*n.pi/alpha * (Y[third_condition] - 1 + alpha/2))) 

    #now multiply both wiindows 
    window = window_X * window_Y
    return window 
    

def rectTukeywindow (size , alpha = 0.5):  #this function will remove all targets outside unit tukey window and then will blur targets with gaussian for fixing targets points according to size of target and for blurring to remove sharp transitions (ringing effect)
    #rectangular tukey window... unity region is defined by alpha..     
    [w,h] = size

    #Window in x direction (vertical direction)
    X = n.arange(w).reshape(w,1)
    X = X / (w-1.0)
    X = X*n.ones((1,h),'d')
    window_X = n.ones(X.shape) 
    # first condition 0 <= x < alpha/2
    first_condition = X<alpha/2
    window_X[first_condition] = 0
    # second condition already taken care of 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X>=(1 - alpha/2)
    window_X[third_condition] = 0     
        
    #Window in horizontal directions 
    Y = n.arange(h).reshape(1,h)
    Y = Y / (h-1.0)
    Y = Y*n.ones((w,1),'d')
    window_Y = n.ones(Y.shape) 
    # first condition 0 <= x < alpha/2
    first_condition = Y<alpha/2
    window_Y[first_condition] = 0
    # second condition already taken care of 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y>=(1 - alpha/2)
    window_Y[third_condition] = 0

    #now multiply both wiindows 
    window = window_X * window_Y
    return window    
    
    
    
    
def meanunit(tile):
        tile = tile - tile.mean()
        length = n.sqrt( (tile*tile).sum() )
        if length > 0.0:
            tile = (1.0/length) * tile
        return tile
   

            
class Classifier:
    
    def __init__(self,size, kfun, kparam, pfun = preprocess, reg=0.01, sigma2 = 2.0 ,alpha = 0.25,  **kwargs):       #sigma is kernal parameter,and reg is regularization paramter (lambda) 
        
        self.originalSize = size        
        self.size = [(int (size[0] / (1-alpha)) +  int(size[0] / (1-alpha)) % 2) , (int (size[1] / (1-alpha)) +  int(size[1] / (1-alpha)) % 2)]                          #for alpha lets say 25%.. To ensurre that size is always even
 
        self.reg=reg
        self.kparam=kparam                                        #this is sigma of Kernel
        self.kfun = kfun
        self.preprocess = pfun
        self.sigma2 = sigma2                                      #this sigma is used in gaussian blurring of target images
        self.num_training = 0
        self.denominator = n.zeros(self.size,n.complex128)          
        self.numerator = n.zeros(self.size,n.complex128)         
        self.test_image_preprocessed = n.zeros(self.size,n.complex128)            
                
        self.alpha = alpha                                    
        self.window=tukeywindow(self.size , self.alpha)                      #tukey window
        self.targetwindow = rectTukeywindow (self.size , self.alpha)    

                         
     

    def Process_test_image(self,tile):    

          test_padded = self.FeaturesSelection (tile)              
          self.test_image_preprocessed = test_padded
          return self.test_image_preprocessed
         
    def ApplyCorrelationFilter(self,tile,output):

                 target = self.TargetPadding (output)   
                 target = self.preprocessTarget (target)       
                 
                 
                 tile_padded = self.FeaturesSelection (tile)
                 

                 k = self.kfun(self.kparam,tile_padded)
              
                 K = n.fft.fft2(k)
 
                 self.denominator += (K + self.reg)          
                 #now processing numerator term
            
                 k = self.kfun(self.kparam, self.test_image_preprocessed , tile_padded)                  
                 
                 K_j = n.fft.fft2(k)
                 
                 G_j = n.fft.fft2 (target)
                 self.numerator += (G_j * K_j)

                 self.num_training = self.num_training + 1                 

                 
    def average(self):
        self.alpha_classifier = self.alpha_classifier / self.num_training



    def resizewindow(self,size):
               if n.shape(self.window) == size:
                   h=self.window 
        
               else:
                   h=tukeywindow(size , self.alpha)
               return h
    

    def resizeTargetwindow(self,size):
               if n.shape(self.targetwindow) == size:
                   h=self.targetwindow 
        
               else:
                   h = rectTukeywindow(size , self.alpha)
               return h
    
    def FeaturesSelection (self, tile):
        
        W = self.resizewindow(self.size)
        pI = np.atleast_3d(self.preprocess (tile))
        
        assert len(np.shape(pI))==3
        tile_pre = []
        for i in range(pI.shape[2]):              
            tile_pre.append( self.EdgesPadding(pI[:,:,i])*W)

        return np.stack(tile_pre,axis = 2)

  
            
    def preprocessTarget (self, target):
            target = target * self.resizeTargetwindow (n.shape(target))         #rectangular tukey  window on target
            #target = self.blur (target)                                         #blurring is to remove rectangular window effects and to make size of dots comparable with actual target sizes            
            return target

            
    def resizefilter(self,size):
                 
            if size == self.alpha_classifier.shape:
                return self.alpha_classifier
            
            else :
                flter = n.fft.ifft2(self.alpha_classifier)
                w,h = size
                
                fw,fh = flter.shape
                tmp = n.zeros((w,h), n.complex128) 
                
                w = min(w,fw)
                h = min(h,fh)
                
                tmp[ :w/2, :h/2] = flter[ :w/2, :h/2]
                tmp[ :w/2,-h/2:] = flter[ :w/2,-h/2:]
                tmp[-w/2:,-h/2:] = flter[-w/2:,-h/2:]
                tmp[-w/2:, :h/2] = flter[-w/2:, :h/2]
                
                self.alpha_classifier = n.fft.fft2(tmp)
            
            return self.alpha_classifier
            
            
    def TestResponse(self):         #return the response immage to the test image  in spatial domain
 
          C = self.numerator / self.denominator              
          response = n.real(n.fft.ifft2(C)) 
          return self.RemovePadding (response)   
                                      #result is in spatial domain
        
    def imagefilter(self):
             img=n.abs(n.fft.fftshift(n.fft.ifft2(n.conj(self.alpha_classifier))))
             return img
    
    def TrainingExamples (self):
        return self.num_training        

    def blur (self, target):                     #sigma 2 for histopathology images large data set
        blurred = gaussian_filter(target, self.sigma2)    #blurr to remove rectangluar window effects and also to make the shapes of dots comparable with actual targets     
        return blurred

        
        
        
    def EdgesPadding (self, tile):                     #Padding with Edges pixels on each side

        img = n.zeros(self.size)                       
        
        [p_rows , p_cols] = self.size                   #Number of Padded Rows and columns
        
        [t_rows, t_cols] = tile.shape                   #Number of Original tile image Rows and Columns
        
        rowStart =  (p_rows - t_rows) / 2
        rowEnd   =  rowStart + t_rows

        colStart = (p_cols - t_cols) / 2
        colEnd   =  colStart + t_cols
        
        img[rowStart : rowEnd , colStart : colEnd] = 1.0*tile                       #just in the middle, place the original image
        
        img [0:rowStart         , colStart : colEnd] =  tile [0,:]                   #Padd top ros
        img [rowEnd : p_rows , colStart : colEnd] =  tile [t_rows - 1 , :]           #padd bottom row
        
        
        temp = n.transpose (img [rowStart : rowEnd  ,       0 : colStart])           #Padd left column
        temp[:,:] = tile [:,0]
        img [rowStart : rowEnd  ,       0 : colStart] = n.transpose (temp)
        
        
        temp = n.transpose (img [rowStart : rowEnd  , colEnd : p_cols])             #Padd Right column
        temp[:,:] = tile [: , t_cols-1]
        img [rowStart : rowEnd  , colEnd : p_cols] = n.transpose (temp)
        
        
        img [0:rowStart         ,         0:colStart] = tile[0,0]                    #padd upper left squared diagnol
        img [0:rowStart         ,    colEnd : p_cols] = tile[0, t_cols-1]              #padd upper Right squared diagnol
        img [rowEnd : p_rows    ,         0:colStart] = tile[t_rows - 1 ,0]               #padd bottom left squared diagnol
        img [rowEnd : p_rows    ,    colEnd : p_cols] = tile[t_rows-1 , t_cols-1]               #padd Bottom Right  squared diagnol
        
        
        return img        
     
     
     
     
     
           
    def TargetPadding (self, target):          #this would padd zeros to target images
        
        img = n.zeros(self.size)                       
                   
            
        [p_rows , p_cols] = self.size                   #Number of Padded Rows and columns
        
        [t_rows, t_cols] = target.shape                   #Number of Original tile image Rows and Columns
        
        rowStart =  (p_rows - t_rows) / 2
        rowEnd   =  rowStart + t_rows

        colStart = (p_cols - t_cols) / 2
        colEnd   =  colStart + t_cols
        
        img[rowStart : rowEnd , colStart : colEnd] = 1.0*target        #just in the middle, place the original image

        return img
        



        
        
    def RemovePadding (self, tile):        #to get the original image of size 500 x 500 back from Padded Image
        

        [p_rows , p_cols] = self.size                   #Number of Padded Rows and columns
        
        [t_rows, t_cols] = self.originalSize                   #Number of Original tile image Rows and Columns
        
        rowStart =  (p_rows - t_rows) / 2
        rowEnd   =  rowStart + t_rows

        colStart = (p_cols - t_cols) / 2
        colEnd   =  colStart + t_cols
      
        img = tile[rowStart : rowEnd , colStart : colEnd]
        
        return img
            
            
            
        

            
            
            