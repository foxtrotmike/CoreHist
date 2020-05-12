


using Images, ImageView, ImageMagick, Colors
using FileIO
function avgGaussianKernel(sigma, x, y = nothing)
    k = zeros(Complex128, size(x)[1:2])
    if length(size(x))!=3
      x=reshape(x, size(x)[1],size(x)[2],1); #eq. to atleast_3d
    end
    for i in range(1 , size(x)[3])
        if y==nothing
            k += dense_gauss_kernel(sigma, x[:,:,i])
        else
            k += dense_gauss_kernel(sigma, x[:,:,i], y[:,:,i])
        end
    end
    k = k / size(x)[3]
    return k

end

function dense_gauss_kernel(sigma, x, y=nothing)
    #=
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must alsoc
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    =#

    xf = fft(x)  # x in Fourier domain
    x_flat = reshape(transpose(x), prod(size(x)))
    xx = *(transpose(x_flat), x_flat)  # squared norm of x

    if y!= nothing
        # general case, x and y are different
        yf = fft(y)
        y_flat = reshape(transpose(y), prod(size(y))) #y.flatten()dense_gauss_kernel
        yy = *(transpose(y_flat), y_flat)
    else
        # auto-correlation of x, avoid repeating a few operations
        yf = xf
        yy = xx
    end

    # cross-correlation term in Fourier domain
    #xyf = dot.(xf, conj(yf))
    xyf = xf.*conj(yf)
    # to spatial domain
    xyf_ifft = ifft(xyf)

    xy = real(xyf_ifft)
    # calculate gaussian response for all positions
    scaling = -1 / (sigma^2)
    xx_yy = xx + yy
    xx_yy_2xy = xx_yy - 2 * xy
    k = exp(scaling * max(0.0, xx_yy_2xy))
    return k
  end #end of dense_gauss_kernel


function preprocess(I)
    #=
    Receives a raw image, it preprocesses and sets the channels
    Input: MxNxD Image
    Output: MxNxP Image
    =#

    I=I[1:300,1:300,:]
    Zg=Gray.(I)
    Zg=reshape(Zg,size(I)[1:2]) #convert does not take 3D matrix
    #Zg=convert(Array{Float32}, raw(Zg))/256 #converting to Integer array

    Zg=Zg[:,:]
    S=imfilter(Zg, Kernel.sobel()) #applying sobel filter
    LG= imfilter(Zg, Kernel.LoG(3.0)) #gaussian_laplace(Zg, 3.0)
    ZZ = cat(3, Zg, S, LG)
    #ZZ=convert(Array{Int32}, raw(ZZ))

    ZZ=meanunit(ZZ)

    return ZZ

  end #end of preprocess

function meanunit(tile)
        tile = tile - mean(tile)
        len = sqrt( sum(.*(tile,tile)))
        if len > 0.0
            tile = (1.0/len) * tile
        end
        return tile
    end


function tukeywindow(mysize, alpha=0.5)
    #=
    The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
    =#
    # Special cases
#    if alpha <= 0:
#        return np.ones(window_length) #rectangular window
#    elif alpha >= 1:
#        return np.hanning(window_length)
    w,h = mysize

    # Normal case
    #Window in x direction (vertical direction)

    X=reshape(range(0,w), w,1)
    X = X / (w-1.0)

    X= .*(X,ones(Float32, (1,h)))
    window_X = ones(size(X))
    # first condition 0 <= x < alpha/2
    first_condition = X .< alpha/2
    window_X[first_condition] = 0.5 * (1 + cos.(2*pi/alpha * (X[first_condition] - alpha/2) ))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X .>= (1 - alpha/2)
    window_X[third_condition] = 0.5 * (1 + cos.(2*pi/alpha * (X[third_condition] - 1 + alpha/2)))

    #Window in horizontal directions
    #Y = n.arange(h).reshape(1,h)
    Y=reshape(range(0,h), 1, h)
    Y = Y / (h-1.0)
    Y = .*(Y, ones(Float32,(w,1)))
    window_Y = ones(size(Y))
    # first condition 0 <= x < alpha/2
    first_condition = Y .< alpha/2
    window_Y[first_condition] = 0.5 * (1 + cos.(2*pi/alpha * (Y[first_condition] - alpha/2) ))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y .>= (1 - alpha/2)
    window_Y[third_condition] = 0.5 * (1 + cos.(2*pi/alpha * (Y[third_condition] - 1 + alpha/2)))

    #now multiply both wiindows
    window = .*(window_X, window_Y)

    return window
 end


function rectTukeywindow(mysize , alpha = 0.5)  #this function will remove all targets outside unit tukey window and then will blur targets with gaussian for fixing targets points according to size of target and for blurring to remove sharp transitions (ringing effect)
    #rectangular tukey window... unity region is defined by alpha..
    w,h= mysize

    #Window in x direction (vertical direction)
    X=reshape(range(0,w), w, 1)
    X = X / (w-1.0)
    X = .*(X, ones(Float32, (1,h)))
    window_X = ones(size(X))
    # first condition 0 <= x < alpha/2
    first_condition = X .< alpha/2
    window_X[first_condition] = 0
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = X .>= (1 - alpha/2)
    window_X[third_condition] = 0

    #Window in horizontal directions
    Y=reshape(range(0,h), 1,h)
    Y = Y / (h-1.0)
    Y = .*(Y, ones(Float32, (w,1)))
    window_Y = ones(size(Y))
    # first condition 0 <= x < alpha/2
    first_condition = Y .< alpha/2
    window_Y[first_condition] = 0
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = Y .>= (1 - alpha/2)
    window_Y[third_condition] = 0

    #now multiply both wiindows
    window = .*(window_X, window_Y)
    return window
  end


type Classifier
        originalSize
        mysize
        reg
        kparam                                        #this is sigma of Kernel
        kfun
        preprocess
        sigma2
        num_training
        denominator
        numerator
        test_image_preprocessed
        alpha
        window
        targetwindow
      #  alpha_classifier
        Classifier()=new()

end


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Tested, works correctly"""
function Process_test_image(obj::Classifier,tile)

  test_padded =FeaturesSelection(obj, tile)
  obj.test_image_preprocessed = test_padded

  return obj.test_image_preprocessed
end

function ApplyCorrelationFilter(obj::Classifier, tile, output)

   target = TargetPadding(obj, output)

   target = preprocessTarget(obj, target)

   tile_padded = FeaturesSelection(obj, tile)

   k = obj.kfun(obj.kparam, tile_padded)

   K = fft(k)

   obj.denominator += (K + obj.reg)

   #now processing numerator term
   k = obj.kfun(obj.kparam, obj.test_image_preprocessed , tile_padded)

   K_j = fft(k)
   G_j = fft(target)
   obj.numerator += G_j.*K_j
   obj.num_training = obj.num_training + 1

end

#=
function average(obj::Classifier)
  obj.alpha_classifier = obj.alpha_classifier/obj.num_training
end
=#

#Tested, works correctly
function resizewindow(obj::Classifier, mysize)
   if size(obj.window) == mysize
       h=obj.window
   else
       h=tukeywindow(mysize , obj.alpha)
  end
  return h
end

function resizeTargetwindow(obj::Classifier, mysize)
   if size(obj.targetwindow) == mysize
       h=obj.targetwindow
   else
       h = rectTukeywindow(mysize , obj.alpha)
     end
   return h
end


#Tested, works correctly
function FeaturesSelection(obj:: Classifier, tile)

    W = resizewindow(obj, obj.mysize)
    pI=obj.preprocess(tile) #outer function, not belong to julia class
    if length(size(pI))!=3
      pI=reshape(pI, size(pI)[1], size(pI)[2], 1)
    end

    #pI = np.atleast_3d(preprocess(obj, tile))
    assert(length(size(pI))==3)
    tile_pre = []
    for i in range(1, size(pI)[3])
      if i==1
        tile_pre=cat(3, .*(EdgesPadding(obj,pI[:,:,i]), W))
      else
        tile_pre=cat(3, tile_pre, .*(EdgesPadding(obj,pI[:,:,i]),W))
      end
    end

    return tile_pre # cat(3, tile_pre)
end






function preprocessTarget(obj::Classifier, target)
    target = .*(target, resizeTargetwindow(obj, size(target)))         #rectangular tukey  window on target
    #target = self.blur (target)                                         #blurring is to remove rectangular window effects and to make size of dots comparable with actual target sizes
    return target
  end

#=
function resizefilter(obj::Classifier,mysize)

      if mysize == size(obj.alpha_classifier)
        return obj.alpha_classifier

      else
          flter = ifft(obj.alpha_classifier)
          w,h = mysize

          fw,fh = size(flter)img
          tmp = zeros(Complex64, (w,h))
          w = min(w,fw)
          h = min(h,fh)

          tmp[ 1:Int(w/2), 1:Int(h/2)] = flter[ 1:Int(w/2), 1:Int(h/2)]
          tmp[ 1:Int(w/2), end-Int(h/2-1):end] = flter[ 1:w/2, end-Int(h/2-1):end]
          tmp[end-Int(w/2):end, end-Int(h/2-1):end] = flter[end-Int(w/2):end, end-Int(h/2-1):end]
          tmp[end-Int(w/2):end, 1:Int(h/2)] = flter[end-Int(w/2):end, end:Int(h/2)]

          obj.alpha_classifier = fft(tmp)
        end
      return obj.alpha_classifier
end
=#


function TestResponse(obj::Classifier)         #return the response immage to the test image  in spatial domain
    C = obj.numerator ./ (obj.denominator)
    response = real(ifft(C))
    return RemovePadding(obj, response)                                     #result is in spatial domain
end


function imagefilter(obj::Classifier)
  img=abs(fftshift(ifft(conj(obj.alpha_classifier))))
  return img
end


function TrainingExamples(obj::Classifier)
    return obj.num_training
end



#ok
function blur(obj::Classifier, target)                     #sigma 2 for histopathology images large data set
    blurred = imfilter(target, Kernel.gaussian(obj.sigma2))    #blurr to remove rectangluar window effects and also to make the shapes of dots comparable with actual targets
    return blurred
end


#ok
function EdgesPadding(obj::Classifier, tile)                   #Padding with Edges pixels on each side

        img = zeros(obj.mysize)
        p_rows , p_cols = obj.mysize                   #Number of Padded Rows and columns
        t_rows, t_cols = size(tile)                   #Number of Original tile image Rows and Columns
        rowStart =  Int(floor(((p_rows - t_rows) / 2)))
        rowEnd   =  Int(floor((rowStart + t_rows)))-1
        colStart = Int(floor(((p_cols - t_cols) / 2)))
        colEnd   =  Int(floor((colStart + t_cols)))-1
        img[rowStart : rowEnd , colStart : colEnd] = 1.0*tile                       #just in the middle, place the original image

        for i in range(1, rowStart)
          img[i, colStart : colEnd] =  tile[1,:]                   #Padd top rows
        end

        for i in range(rowEnd, p_rows-rowEnd)
          img[i , colStart : colEnd] =  tile[t_rows - 1 , :]           #padd bottom row
        end

        temp = transpose(img[rowStart : rowEnd, 1: colStart])           #Padd left column

        for i in range(1, rowStart) #due to transpose range is upto rowStart
          temp[i,:] = tile[:,1]
        end
        img[rowStart : rowEnd  ,  1 : colStart] = transpose(temp)

        for i in range(colEnd, p_cols-colEnd)
          img[rowStart : rowEnd  , i] = tile[:,t_cols]#transpose(temp)
        end
        img[1:rowStart , 1:colStart] = tile[1,1]                    #padd upper left squared diagnol
        img[1:rowStart , colEnd : p_cols] = tile[1, t_cols-1]              #padd upper Right squared diagnol
        img[rowEnd : p_rows ,1:colStart] = tile[t_rows - 1 ,1]               #padd bottom left squared diagnol
        img[rowEnd : p_rows, colEnd : p_cols] = tile[t_rows-1 , t_cols-1]               #padd Bottom Right  squared diagnol

        return img
end


#Working ok
function TargetPadding(obj::Classifier, target)        #this would padd zeros to target images
        img = zeros(obj.mysize)
        p_rows , p_cols = obj.mysize                 #Number of Padded Rows and columns
        t_rows, t_cols = size(target)                   #Number of Original tile image Rows and Columns
        rowStart =  Int((p_rows - t_rows) / 2)
        rowEnd   =  Int(rowStart + t_rows)-1
        colStart = Int((p_cols - t_cols) / 2)
        colEnd   =  Int(colStart + t_cols-1)
        img[rowStart : rowEnd , colStart : colEnd] = 1.0*target        #just in the middle, place the original image
        return img
end


#working, OK
function RemovePadding(obj::Classifier, tile)       #to get the original image of size 500 x 500 back from Padded Image
        p_rows , p_cols = obj.mysize                   #Number of Padded Rows and columns
        t_rows, t_cols = obj.originalSize                   #Number of Original tile image Rows and Columns
        rowStart =  Int((p_rows - t_rows) / 2)
        rowEnd   =  Int(rowStart + t_rows)-1
        colStart = Int((p_cols - t_cols) / 2)
        colEnd   =  Int(colStart + t_cols)-1
        img = tile[rowStart : rowEnd , colStart : colEnd]
        return img
end
