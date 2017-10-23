import cv2
import numpy as np
import math
from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="\t- Time Elapsed: %s s"):
    print(fmt % (time() - _tstart_stack.pop()))

classic = np.array([[0.567, 0.635, 0.608, 0.514, 0.424, 0.365, 0.392, 0.486],
                    [0.847, 0.878, 0.910, 0.698, 0.153, 0.122, 0.090, 0.302],
                    [0.820, 0.969, 0.941, 0.667, 0.180, 0.031, 0.059, 0.333],
                    [0.725, 0.788, 0.757, 0.545, 0.275, 0.212, 0.243, 0.455],
                    [0.424, 0.365, 0.392, 0.486, 0.567, 0.635, 0.608, 0.514],
                    [0.153, 0.122, 0.090, 0.302, 0.847, 0.878, 0.910, 0.698],
                    [0.180, 0.031, 0.059, 0.333, 0.820, 0.969, 0.941, 0.667],
                    [0.275, 0.212, 0.243, 0.455, 0.725, 0.788, 0.757, 0.545]])
bayer5 = np.array([[0.513, 0.272, 0.724, 0.483, 0.543, 0.302, 0.694, 0.453],
                   [0.151, 0.755, 0.091, 0.966, 0.181, 0.758, 0.121, 0.936],
                   [0.634, 0.392, 0.574, 0.332, 0.664, 0.423, 0.604, 0.362],
                   [0.060, 0.875, 0.211, 0.815, 0.030, 0.906, 0.241, 0.845],
                   [0.543, 0.302, 0.694, 0.453, 0.513, 0.272, 0.724, 0.483],
                   [0.181, 0.758, 0.121, 0.936, 0.151, 0.755, 0.091, 0.966],
                   [0.664, 0.423, 0.604, 0.362, 0.634, 0.392, 0.574, 0.332],
                   [0.030, 0.906, 0.241, 0.845, 0.060, 0.875, 0.211, 0.815]])

def ordered_dithering(input, kernel):
    img = input.copy()
    step1 = kernel.shape[0]
    step2 = kernel.shape[1]

    total = 0
    for i in range(0, img.shape[0], step1):
        for j in range(0, img.shape[1], step2):
            idx1 = i
            for k in range(0, kernel.shape[0]):
                idx2 = j
                for l in range(0, kernel.shape[1]):
                    if(idx1<img.shape[0] and idx2<img.shape[1]):
                        if (img[idx1, idx2] > kernel[k, l]):
                            img[idx1, idx2] = 255
                        else:
                            img[idx1, idx2] = 0
                    idx2 = idx2+1
                idx1 = idx1+1
    print(total)
    return img

def manual_convolution(image, kernel):
    print('\t- Manual Convolution')
    tic()
    img = image.copy()
    m = int(kernel.shape[0]/2)
    imout = np.zeros((img.shape[1]+(2*m),img.shape[0]+(2*m)), dtype=np.int)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            imout[i+m,j+m] = img[i,j]
    x = 0
    for i in range(m, imout.shape[0]-m):
        y = 0
        for j in range(m,imout.shape[1]-m):
            total = 0
            idi=-m
            for k in range(0,kernel.shape[0]):
                idj=-m
                for l in range(0,kernel.shape[1]):
                    total = total+(float(imout[i+idi,j+idj])*float(kernel[k,l]))
                    idj+=1
                idi+=1
            img[x,y] = int(total)
            y+=1
        x+=1
    toc()
    return img

def gaussian_kernel(l=7, sig=1.3):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    print('\t- Getting Gaussian Kernel')
    tic()
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    toc()
    return kernel / np.sum(kernel)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def single_run(impath, classic, bayer5, show):
    image = cv2.imread(impath+'.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    classic = classic*255
    bayer5 = bayer5*255
    gray1 = ordered_dithering(gray, classic)
    gray2 = ordered_dithering(gray, bayer5)

    cv2.imwrite(impath + '_classical4.png', gray1)
    cv2.imwrite(impath + '_bayer5.png', gray2)

    filter3 = gaussian_kernel(11, 1.3)
    result3 = manual_convolution(gray, filter3)
    blur1 = manual_convolution(gray1, filter3)
    blur2 = manual_convolution(gray2, filter3)

    psnr_c4 = psnr(gray, gray1)
    psnr_b5 = psnr(gray, gray2)

    print('PSNR')
    print('Classical-4 got PSNR\t= {}'.format(psnr_c4))
    print('Bayer-5 got PSNR\t= {}'.format(psnr_b5))

    hpsnr_c4 = psnr(result3, blur1)
    hpsnr_b5 = psnr(result3, blur2)

    print('HPSNR')
    print('Classical-4 got HPSNR\t= {}'.format(hpsnr_c4))
    print('Bayer-5 got HPSNR\t= {}'.format(hpsnr_b5))

    if show == 1:
        cv2.imshow('Original', gray)
        cv2.imshow('Classical 4', gray1)
        cv2.imshow('Bayer 5', gray2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

single_run('Lenna', classic, bayer5, 0)
single_run('boat', classic, bayer5, 0)