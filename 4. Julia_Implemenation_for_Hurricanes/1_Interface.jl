
#FFTW.set_num_threads(4)
using Images, ImageView, ImageMagick, Colors
using FileIO
obj=Classifier()

function initiate(obj:: Classifier,mysize, kfun, kparam, pfun=preprocess, reg=0.0001, sigma2=5.0 , alpha=0.25)
  obj.originalSize = mysize
  obj.mysize=((Int(floor(mysize[1] / (1-alpha))) +  Int(floor((mysize[1] / (1-alpha))) % 2)) , Int(floor((mysize[2] / (1-alpha)))) +  Int(floor((mysize[2] / (1-alpha)) % 2)))
  obj.reg=reg
  obj.kparam=kparam                                        #this is sigma of Kernel
  obj.kfun = kfun
  obj.preprocess = pfun
  obj.sigma2 = sigma2                                      #this sigma is used in gaussian blurring of target images
  obj.num_training = 0
  obj.denominator = zeros(Complex64, obj.mysize)
  obj.numerator = zeros(Complex64,obj.mysize)
  obj.test_image_preprocessed = zeros(Complex64, obj.mysize)
  obj.alpha = alpha
  obj.window=tukeywindow(obj.mysize , obj.alpha)            #tukey window
  obj.targetwindow = rectTukeywindow(obj.mysize , obj.alpha)
end

sigma=3.0
msize=(300,300)
initiate(obj,msize, avgGaussianKernel, sigma)

output=zeros(300,300)
output[150,150]=1.0
output=imfilter(output, Kernel.gaussian(5.0));

savePath="/home/bismillah/juliaOutput/"
tpath="/home/bismillah/Hurricane/KATRINA/"
tfiles=readdir(tpath)
allPath="/home/bismillah/Hurricane/Combine_Data/"
#KATRINA excluded
folders=readdir(allPath)


tic()
for i in range(1, length(tfiles)-36)

  timg=load(string(tpath,tfiles[i]))

  Process_test_image(obj,timg)
  for j in range(1, length(folders)-93)
    singleFolder=string(allPath, folders[j],"/")
    files=readdir(singleFolder)

    for k in range(1, length(files))

        img=load(string(singleFolder,files[k]))
        ApplyCorrelationFilter(obj, img, output)

    end

  end
#  tr=TestResponse(obj)

#  save(string(savePath, i,".png"), tr)
#  imshow(tr)
end
toc()


#=
tic();
for i in range(1, length(tfiles)-36)
  timg=load(string(tpath,tfiles[i]))
  Process_test_image(obj,timg)
  singleFolder=string(allPath, "ALEX/")
  files=readdir(singleFolder)
  ApplyCorrelationFilter(obj, timg, output)
  tr=TestResponse(obj)
  #save(string(savePath, j,".png"), tr)
  imshow(tr)
end
toc()
=#





#=
function cal(img)
  return maximum(real(img)), minimum(real(img)), sum(real(img))
end
Process_test_image(obj,timg);
k = ApplyCorrelationFilter(obj, timg, output);
print(k)
  #tr=TestResponse(obj)
  #imshow(tr)
=#
#=
myimg="/home/bismillah/Hurricane/Combine_Data/ALEX/2004214N30282.ALEX.2004.08.03.0900.39.GOE-12.068.hursat-b1.v05.nc.png"
timg=load(myimg)
tt=Process_test_image(obj,transpose(timg));
imshow(tt)

testImg=transpose(reshape(range(0,100),10,10))
sigma=3.0
msize=(10,10)
initiate(obj,msize, avgGaussianKernel, sigma)
testImg=convert(Array{Float32}, testImg)
temp=Process_test_image(obj,testImg)
imshow(temp)

imgp="/home/bismillah/Hurricane/Combine_Data/ALBERTO/2006161N20275.ALBERTO.2006.06.10.0900.27.GOE-12.024.hursat-b1.v05.nc.png"
timg=load(imgp)
pti=Process_test_image(obj, timg)
imshow(transpose(pti[:,:,1]))
=#
#=
sm=0.0
for i in range(1, 100)
temp=rand(600,600)
tic()
fft(temp)
t=toc()
sm=sm+t
end
sm/100
=#
