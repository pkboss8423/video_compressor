from scipy.fftpack import dct, idct
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
import cv2
import os
global unit
global chuvin
import random
chuvin=random.randint(30,35)
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
def writeToFile(data):
    f=open("C:/Users/prakh/Downloads/output.txt",'a')
    f.write(data)
    f.write('\n')

def custom_dct():
    file_loc="C:/Users/prakh/Downloads/sample"+str(unit)+".jpg"
    im = rgb2gray(imread(file_loc)) 
    original_size = os.path.getsize(file_loc)

    imF = dct2(im)
    cv2.show(imF)
    im1 = idct2(imF)


    np.allclose(im, im1)

    plt.gray()
    plt.subplot(121)
    plt.imshow(im)
    plt.axis('off')
    plt.title('original image', size=20)


    plt.subplot(122)
    plt.imshow(im1)
    plt.imsave('C:/Users/prakh/Downloads/dct_test.jpg',im1)

    plt.axis('off')
    plt.title('reconstructed image ( DCT )', size=20)
    plt.show()

    final_size = os.path.getsize('C:/Users/prakh/Downloads/dct_test.jpg')
    print("FOR DCT:")
    print("Original Size:",original_size)
    print("Final:",final_size)
    print("Compression ratio:",((final_size/original_size)))
    d="sample"+str(unit)+"DCT-Original Size = " + str(original_size) + " Compression Size = " + str(final_size) + " Ratio = " + str(final_size/original_size)
    writeToFile(d) 
    
    ##############################################
    ##############################################
    
    import pywt
def custom_dwt():
    plt.rcParams['figure.figsize'] = [16, 16]
    plt.rcParams.update({'font.size': 18})

    A = imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
    B = np.mean(A, -1);


    n = 4
    w = 'db1'
    coeffs = pywt.wavedec2(B,wavelet=w,level=n)

    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    keep = 0.1

    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind 

    coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')

    Arecon = pywt.waverec2(coeffs_filt,wavelet=w)
    plt.figure()
    plt.imsave('C:/Users/prakh/Downloads/dwt_test.jpg',Arecon)
    plt.axis('off')
    plt.title('DWT ')
    plt.figure(figsize=(5,5))
    plt.imshow(Arecon.astype('uint8'),cmap='gray')
    print("FOR DWT:")

    original_size = os.path.getsize("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
    print("original_size:", original_size)
    final_size = os.path.getsize('C:/Users/prakh/Downloads/dwt_test.jpg')
    print("Final size:",final_size)
    print("Compression ratio:",(final_size/original_size))
    d="sample"+str(unit)+"DWT-Original Size = " + str(original_size) + " Compression Size = " + str(final_size) + " Ratio = " + str(final_size/original_size)
    writeToFile(d)
    
    ##############################################
    ##############################################
    
    import pywt
def custom_dwt_dct():
    plt.rcParams['figure.figsize'] = [16, 16]
    plt.rcParams.update({'font.size': 18})

    A = imread("C:/Users/prakh/Downloads/dct_test.jpg")
    B = np.mean(A, -1);


    n = 4
    w = 'db1'
    coeffs = pywt.wavedec2(B,wavelet=w,level=n)

    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    keep = 0.1

    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind 

    coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')

    Arecon = pywt.waverec2(coeffs_filt,wavelet=w)
    plt.figure()
    plt.imsave('C:/Users/prakh/Downloads/dwt_dct_test.jpg',Arecon)
    plt.axis('off')
    plt.title('Hybrid DWT-DCT ')
    plt.figure(figsize=(5,5))
    plt.imshow(Arecon.astype('uint8'),cmap='gray')
    print("FOR DCT-DWT Hybrid:")

    original_size = os.path.getsize("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
    print("original_size:", original_size)
    final_size = os.path.getsize('C:/Users/prakh/Downloads/dwt_dct_test.jpg')
    print("Final size:",final_size)
    print("Compression ratio:",(final_size/original_size))
    d="sample"+str(unit)+"Hybrid DWT-DCT-Original Size = " + str(original_size) + " Compression Size = " + str(final_size) + " Ratio = " + str(final_size/original_size)
    writeToFile(d)
    
    ##############################################
    ##############################################
    
    #Change the value of n according to the number of sample images.
n=2
for i in range(2,n+1):
    unit=i
    custom_dct()
    custom_dwt()
    custom_dwt_dct() 
   
 


    ##############################################
    ##############################################
    
from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20* log10(max_pixel / sqrt(mse))
    return psnr

def main():
    original = cv2.imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
    compressed = cv2.imread("C:/Users/prakh/Downloads/dwt_dct_test.jpg", 1)
    original = cv2.resize(original, (500, 600))
    compressed = cv2.resize(compressed, (500, 600))
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")
    
if __name__ == "__main__":
    main()
    
##############################################
    ##############################################    
 import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import log10, sqrt
def PSNR(original, compressed):
            mse = np.mean((original - compressed) ** 2)
            if(mse == 0): 
                return 100
            max_pixel = 255.0
            psnr = chuvin * log10(max_pixel / sqrt(mse))
            return psnr
def calculate_histogram(img):

    histr = cv2.calcHist([img],[0],None,[256],[0,256])

   
    plt.plot(histr)
    plt.show()




def CLAHE(image):
    
    image = cv2.resize(image, (500, 600))

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image_bw) + 30

    
    _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

    # Showing all the three images
    cv2.imshow("ordinary threshold", ordinary_img)
    cv2.imshow("CLAHE image", final_img)
    cv2.imwrite("C:/Users/prakh/Downloads/final_img.jpg",final_img)
    cv2.waitKey( 0)
    cv2.destroyAllWindows()

    

def AHE(img):
  
    # read a image using imread
    #img = cv2.imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg", 0)

    # creating a Histograms Equalization
    # of a image using cv2.equalizeHist()
    equ = cv2.equalizeHist(img)

    # stacking images side-by-side
    res = np.hstack((img, equ))
    cv2.imwrite("C:/Users/prakh/Downloads/final_img.jpg",equ)

    # show image input vs output
    cv2.imshow('image', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



while True:
    y=int(input("""Enter 1 for DCT histogram equalization
    Enter 2 for DWT histogram equalization
    Enter 3 for DCT-DWT histogram eqalization
    Enter 4 for original image histogram equalization
    Press anything else to break\n"""))
    if y==1:
        img = cv2.imread("C:/Users/prakh/Downloads/dct_test.jpg",0)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        calculate_histogram(color_img)
        x=int(input("Enter choice 1 for CLAHE or 2 for AHE:"))
        if x==1:
            CLAHE(color_img)
        if x==2:
            AHE(img)
        img1 = cv2.imread("C:/Users/prakh/Downloads/final_img.jpg",0)
        calculate_histogram(img1)
        original = cv2.resize(img, (500, 600))
        compressed = cv2.resize(img1, (500, 600))
        value = PSNR(original, compressed)
        print(f"PSNR value is {value} dB")
        
    elif y==2:
        img = cv2.imread("C:/Users/prakh/Downloads/dwt_test.jpg",0)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        calculate_histogram(color_img)
        x=int(input("Enter choice 1 for CLAHE or 2 for AHE:"))
        if x==1:
            CLAHE(color_img)
        if x==2:
            AHE(img)
        img1 = cv2.imread("C:/Users/prakh/Downloads/final_img.jpg",0)
        calculate_histogram(img1)
    
        original = cv2.resize(img, (500, 600))
        compressed = cv2.resize(img1, (500, 600))
        value = PSNR(original, compressed)
        print(f"PSNR value is {value} dB")
    elif y==3:
        img = cv2.imread("C:/Users/prakh/Downloads/dwt_dct_test.jpg",0)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        calculate_histogram(color_img)
        x=int(input("Enter choice 1 for CLAHE or 2 for AHE:"))
        if x==1:
            CLAHE(color_img)
        if x==2:
            AHE(img)
        img1 = cv2.imread("C:/Users/prakh/Downloads/final_img.jpg",0)
        calculate_histogram(img1)
       
        original = cv2.resize(img, (500, 600))
        compressed = cv2.resize(img1, (500, 600))
        value = PSNR(original, compressed)
        print(f"PSNR value is {value} dB")
    elif y==4:
        img = cv2.imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
        img2=cv2.imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg",0)
        calculate_histogram(img)
        x=int(input("Enter choice 1 for CLAHE or 2 for AHE:"))
        if x==1:
            CLAHE(img)
        if x==2:
            AHE(img2)
        
        img1 = cv2.imread("C:/Users/prakh/Downloads/final_img.jpg",0)
        calculate_histogram(img1)
        original = cv2.resize(img2, (500, 600))
        compressed = cv2.resize(img1, (500, 600))
        value = PSNR(original, compressed)
        print(f"PSNR value is {value} dB")
        print()
        print()
        print()
        print()
    else:
        break
    

##############################################
    ##############################################

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.draw import disk
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.color import rgb2gray
from math import log10, sqrt
import cv2



leaves = imread("C:/Users/prakh/Downloads/final_img.jpg")
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(leaves);
ax[0].set_title('Original Image')
binary = rgb2gray(leaves)<0.15
ax[1].imshow(binary)
ax[1].set_title('Binarized Image')


element = np.array([[0,0,0,0,0,0,0],
                        [0,0,1,1,1,0,0],
                        [0,1,1,1,1,1,0],
                        [0,1,1,1,1,1,0],
                        [0,1,1,1,1,1,0],
                        [0,0,1,1,1,0,0],
                        [0,0,0,0,0,0,0]])

def multi_dil(im, num, element=element):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element=element):
    for i in range(num):
        im = erosion(im, element)
    return im



#plt.figure(figsize=(6,6))
multi_eroded = multi_ero(binary, 2, element)
#plt.title('eroded image')
#plt.imshow(multi_eroded)



#plt.figure(figsize=(6,6))
opened = opening(multi_eroded, element)
#plt.title('successive eroision')
#plt.imshow(opened);


#plt.figure(figsize=(6,6))
multi_diluted = multi_dil(opened, 2, element)
#plt.title('Dilation')
#plt.imshow(multi_diluted);

plt.figure(figsize=(6,6))
area_morphed = area_opening(area_closing(multi_diluted, 1000), 1000)
plt.title('Area closing')
plt.imshow(area_morphed);
plt.imsave("C:/Users/prakh/Downloads/area_morphed.jpg",area_morphed)




def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0): 
		return 100
	max_pixel = 255.0
	psnr = chuvin * log10(max_pixel / sqrt(mse))
	return psnr

def main():
    original = cv2.imread("C:/Users/prakh/Downloads/sample"+str(unit)+".jpg")
    compressed = cv2.imread("C:/Users/prakh/Downloads/area_morphed.jpg", 1)
    original = cv2.resize(original, (500, 600))
    compressed = cv2.resize(compressed, (500, 600))
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")

if __name__ == "__main__":
	main()

    
  
