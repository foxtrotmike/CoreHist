# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:54:49 2016

@author: Asif Ahmad
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:31:29 2016

@author: Asif Ahmad
"""


import numpy as n
from skimage.io import imread, imshow
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter    #for blurring target that are have sharp transitions


def cosinewindow(size):
    w,h = size
    X = n.arange(w).reshape(w,1)
    Y = n.arange(h).reshape(1,h)
    X = X*n.ones((1,h),'d')
    Y = Y*n.ones((w,1),'d')
    
    window = (n.sin(n.pi*X/(w-1.0)))*(n.sin(n.pi*Y/(h-1.0)))
    return window
    


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
               
class MOSSE:
    
    def __init__(self,size, reg=0.01, sigma2=2.0, alpha = 0.5, **kwargs):
        self.originalSize = size        
        self.size = [(int (size[0] / (1-alpha)) +  int(size[0] / (1-alpha)) % 2) , (int (size[1] / (1-alpha)) +  int(size[1] / (1-alpha)) % 2)]                          #for alpha lets say 25%.. To ensurre that size is always even
        self.reg=reg
        self.sigma2=sigma2
        self.num_training = 0
        self.filter= n.zeros(self.size,n.complex128)
        self.numF=n.zeros(self.size,n.complex128)
        self.denF=n.zeros(self.size,n.complex128)

        self.alpha = alpha 
        self.window=tukeywindow(self.size , self.alpha)                      #tukey window
        self.targetwindow = rectTukeywindow (self.size , self.alpha)    

     
        
        
    def addtraining(self,tile,output):
                 g = self.TargetPadding (output)   
                 g = self.preprocessTarget (g)       

                 f = self.preprocess (tile)                 
                 f = self.EdgesPadding (f) * self.resizewindow(self.size)

                 F = n.fft.fft2(f)
                 G = n.fft.fft2(g)
                 cF = n.conj(F)
                 self.numF+=G*cF
                 self.denF+= F*cF+self.reg
                 self.filter = self.numF / (self.denF )
                 
                 self.num_training = self.num_training + 1


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

        
               return h
    def preprocess(self,tile):
            tile = (tile / 255.0)**2
            tile=meanunit(tile)
            return tile


    def preprocessTarget (self, target):
            target = target * self.resizeTargetwindow (n.shape(target))         #rectangular tukey  window on target
            target = self.blur (target)                         #blurring is to remove rectangular window effects and to make size of dots comparable with actual target sizes            
            return target


    def resizefilter(self,size):
                 
            if size == self.filter.shape:
                return self.filter
            
            else :
                flter = n.fft.ifft2(self.filter)
                w,h = size
                
                fw,fh = flter.shape
                tmp = n.zeros((w,h), n.complex128) 
                
                w = min(w,fw)
                h = min(h,fh)
                
                tmp[ :w/2, :h/2] = flter[ :w/2, :h/2]
                tmp[ :w/2,-h/2:] = flter[ :w/2,-h/2:]
                tmp[-w/2:,-h/2:] = flter[-w/2:,-h/2:]
                tmp[-w/2:, :h/2] = flter[-w/2:, :h/2]
                
                self.filter = n.fft.fft2(tmp)
            
            return self.filter
    def correlate(self,tile):
          f = self.preprocess (tile)    
          f = self.EdgesPadding (f) * self.resizewindow(self.size)

          F = n.fft.fft2(f)
          G = self.resizefilter(n.shape(F)) * F

          g = n.fft.ifft2(G)
          cor = n.real(g)
          return self.RemovePadding (cor) 
        
    def imagefilter(self):
             img=n.abs(n.fft.fftshift(n.fft.ifft2(n.conj(self.filter))))
             #img = n.real(n.fft.ifft2(self.filter))
             img = self.RemovePadding(img)
             return img
    
    def conjugate(self):
        return n.conjugate(self.filter)
        
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
            
            
            
            