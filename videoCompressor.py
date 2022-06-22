import sys
import argparse
from scipy.fftpack import dct, idct
import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import shutil
import pywt
import re
import sys
import webcolors
import time
from os.path import isfile, join

def frame_rate(cam):
    _, fo = cam.read()
    framei = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
    bg_avg = np.float32(framei)
    video_width = int(cam.get(3))
    video_height = int(cam.get(4))
    fr = int(cam.get(5))
    print("frame rate of stored video:::",fr)
    return fr

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    sorted_images = sorted(files, key=natural_sort_key)
    for i in range(len(sorted_images)):
        filename=pathIn + sorted_images[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    print("Video Finished")
    out.release()


    

def custom_dwt(A):
    plt.rcParams['figure.figsize'] = [16, 16]
    plt.rcParams.update({'font.size': 18})

    
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
    return Arecon

   
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def custom_dct(a):
    imF = dct2(a)
    im1 = idct2(imF)
    np.allclose(a, im1)
    return im1

def custom_dwt_dct(A):
    x=custom_dct(A)
    y=custom_dwt(x)
    return y

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    x=int(input("""Enter 1 for DCT Compression:
    Enter 2 for DWT Compression:
    Enter 3 for Hybrid DCT-DWT:"""))
    while success and x==2:
        success,image = vidcap.read()
        if success == False:
            break
        
        
        image=custom_dwt(image)
        
        cv2.imwrite( pathOut + "\\%d.jpg" % count, image)     # save frame as JPEG file
        count+=1
    while success and x==1:
        success,image = vidcap.read()
        if success == False:
            break
        
        
        image=custom_dct(image)
        
        cv2.imwrite( pathOut + "\\%d.jpg" % count, image)     # save frame as JPEG file
        count+=1
    while success and x==3:
        success,image = vidcap.read()
        if success == False:
            break
        
        
        image=custom_dwt_dct(image)
        
        cv2.imwrite( pathOut + "\\%d.jpg" % count, image)     # save frame as JPEG file
        count+=1
    print("Frame Breaking Completed,total frames=",count)

if __name__=="__main__":
    pathIn='C:/Users/prakh/Downloads/Video/sample1.mp4'
    pathOut="C:/Users/prakh/Downloads/Video/New folder"
    
    for filename in os.listdir(pathOut):
        file_path = os.path.join(pathOut, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    extractImages(pathIn, pathOut)
    pathIn= 'C:/Users/prakh/Downloads/Video/New folder/'
    pathOut = 'C:/Users/prakh/Downloads/Video/video1.mp4'
    cam=cv2.VideoCapture('C:/Users/prakh/Downloads/Video/sample1.mp4')
    fps = frame_rate(cam)
    convert_frames_to_video(pathIn, pathOut, fps)
