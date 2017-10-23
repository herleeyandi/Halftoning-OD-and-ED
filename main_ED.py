import cv2
import numpy as np
import math
from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="\t- Time Elapsed: %s s"):
    print(fmt % (time() - _tstart_stack.pop()))


def apply_threshold(value):
    "Returns 0 or 255 depending where value is closer"
    return 255 * math.floor(value / 128)

def floyd_steinberg_dither(image):
    print('====Floyd Steinberg====')
    tic()
    old_h, old_w = image.shape
    new_img = np.zeros((old_h+1, old_w+2),dtype=np.uint8)
    new_img[0:old_h, 1:old_w+1] = image.copy()

    x_lim, y_lim = new_img.shape

    for y in range(1, y_lim):
        for x in range(1, x_lim):
            oldpixel = new_img[x, y]

            newpixel = apply_threshold(oldpixel)

            new_img[x, y] = newpixel

            error = oldpixel - newpixel

            if x < x_lim - 1:
                tmp = new_img[x + 1, y] + round(error * 7 / 16)
                new_img[x + 1, y] = tmp

            if x > 1 and y < y_lim - 1:
                tmp = new_img[x - 1, y + 1] + round(error * 3 / 16)
                new_img[x - 1, y + 1] = tmp

            if y < y_lim - 1:
                tmp = new_img[x, y + 1] + round(error * 5 / 16)
                new_img[x, y + 1] = tmp

            if x < x_lim - 1 and y < y_lim - 1:
                tmp = new_img[x + 1, y + 1] + round(error * 1 / 16)
                new_img[x + 1, y + 1] = tmp
    toc()
    return new_img[0:old_h, 1:old_w+1]

def jarvis_dither(image):
    print('====Jarvis====')
    tic()
    old_h, old_w = image.shape
    new_img = np.zeros((old_h + 2, old_w + 4), dtype=np.uint8)
    new_img[0:old_h, 2:old_w + 2] = image.copy()
    x_lim, y_lim = new_img.shape

    for y in range(2, y_lim):
        for x in range(2, x_lim):
            oldpixel = new_img[x, y]

            newpixel = apply_threshold(oldpixel)

            new_img[x, y] = newpixel

            error = oldpixel - newpixel

            if x < x_lim - 1:
                tmp = new_img[x + 1, y] + round(error * 7 / 48)
                new_img[x + 1, y] = tmp

            if x < x_lim - 2:
                tmp = new_img[x + 2, y] + round(error * 5 / 48)
                new_img[x + 2, y] = tmp

            if x > 2 and y < y_lim - 1:
                tmp = new_img[x - 2, y + 1] + round(error * 3 / 48)
                new_img[x - 2, y + 1] = tmp

            if x > 1 and y < y_lim - 1:
                tmp = new_img[x - 1, y + 1] + round(error * 5 / 48)
                new_img[x - 1, y + 1] = tmp

            if y < y_lim - 1:
                tmp = new_img[x, y + 1] + round(error * 7 / 48)
                new_img[x, y + 1] = tmp

            if x < x_lim - 1 and y < y_lim - 1:
                tmp = new_img[x + 1, y + 1] + round(error * 5 / 48)
                new_img[x + 1, y + 1] = tmp

            if x < x_lim - 2 and y < y_lim - 1:
                tmp = new_img[x + 2, y + 1] + round(error * 3 / 48)
                new_img[x + 2, y + 1] = tmp

            if x > 2 and y < y_lim - 2:
                tmp = new_img[x - 2, y + 2] + round(error * 1 / 48)
                new_img[x - 2, y + 2] = tmp

            if y < y_lim - 2:
                tmp = new_img[x - 1, y + 2] + round(error * 3 / 48)
                new_img[x - 1, y + 2] = tmp

            if y < y_lim - 2:
                tmp = new_img[x, y + 2] + round(error * 5 / 48)
                new_img[x, y + 2] = tmp

            if x < x_lim - 1 and y < y_lim - 2:
                tmp = new_img[x + 1, y + 2] + round(error * 3 / 48)
                new_img[x + 1, y + 2] = tmp

            if x < x_lim - 2 and y < y_lim - 2:
                tmp = new_img[x + 2, y + 2] + round(error * 1 / 48)
                new_img[x + 2, y + 2] = tmp
    toc()
    return new_img[0:old_h, 2:old_w + 2]

def stucki_dither(image):
    print('====Stucki====')
    tic()
    old_h, old_w = image.shape
    new_img = np.zeros((old_h + 2, old_w + 4), dtype=np.uint8)
    new_img[0:old_h, 2:old_w + 2] = image.copy()

    x_lim, y_lim = new_img.shape

    for y in range(2, y_lim):
        for x in range(2, x_lim):
            oldpixel = new_img[x, y]

            newpixel = apply_threshold(oldpixel)

            new_img[x, y] = newpixel

            error = oldpixel - newpixel

            if x < x_lim - 1:
                tmp = new_img[x + 1, y] + round(error * 8 / 42)
                new_img[x + 1, y] = tmp

            if x < x_lim - 2:
                tmp = new_img[x + 2, y] + round(error * 4 / 42)
                new_img[x + 2, y] = tmp

            if x > 2 and y < y_lim - 1:
                tmp = new_img[x - 2, y + 1] + round(error * 2 / 42)
                new_img[x - 2, y + 1] = tmp

            if x > 1 and y < y_lim - 1:
                tmp = new_img[x - 1, y + 1] + round(error * 4 / 42)
                new_img[x - 1, y + 1] = tmp

            if y < y_lim - 1:
                tmp = new_img[x, y + 1] + round(error * 8 / 42)
                new_img[x, y + 1] = tmp

            if x < x_lim - 1 and y < y_lim - 1:
                tmp = new_img[x + 1, y + 1] + round(error * 4 / 42)
                new_img[x + 1, y + 1] = tmp

            if x < x_lim - 2 and y < y_lim - 1:
                tmp = new_img[x + 2, y + 1] + round(error * 2 / 42)
                new_img[x + 2, y + 1] = tmp

            if x > 2 and y < y_lim - 2:
                tmp = new_img[x - 2, y + 2] + round(error * 1 / 42)
                new_img[x - 2, y + 2] = tmp

            if y < y_lim - 2:
                tmp = new_img[x - 1, y + 2] + round(error * 2 / 42)
                new_img[x - 1, y + 2] = tmp

            if y < y_lim - 2:
                tmp = new_img[x, y + 2] + round(error * 4 / 42)
                new_img[x, y + 2] = tmp

            if x < x_lim - 1 and y < y_lim - 2:
                tmp = new_img[x + 1, y + 2] + round(error * 2 / 42)
                new_img[x + 1, y + 2] = tmp

            if x < x_lim - 2 and y < y_lim - 2:
                tmp = new_img[x + 2, y + 2] + round(error * 1 / 42)
                new_img[x + 2, y + 2] = tmp
    toc()
    return new_img[0:old_h, 2:old_w + 2]

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

def single_run(impath, show):
    image = cv2.imread(impath + '.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    filter3 = gaussian_kernel(11, 1.3)

    result3 = manual_convolution(gray, filter3)
    gray1 = floyd_steinberg_dither(gray)
    blur1 = manual_convolution(gray1, filter3)
    gray2 = jarvis_dither(gray)
    blur2 = manual_convolution(gray2, filter3)
    gray3 = stucki_dither(gray)
    blur3 = manual_convolution(gray3, filter3)

    cv2.imwrite(impath + '_floyd_steinberg.png', gray1)
    cv2.imwrite(impath + '_jarvis.png', gray2)
    cv2.imwrite(impath + '_stucki.png', gray3)

    psnr_floyd = psnr(gray, gray1)
    psnr_jarvis = psnr(gray, gray2)
    psnr_stucki = psnr(gray, gray3)

    print('PSNR')
    print('Floyd Steinberg got PSNR\t= {}'.format(psnr_floyd))
    print('Jarvis got PSNR\t= {}'.format(psnr_jarvis))
    print('Stucki got PSNR\t= {}'.format(psnr_stucki))

    hpsnr_floyd = psnr(result3, blur1)
    hpsnr_jarvis = psnr(result3, blur2)
    hpsnr_stucki = psnr(result3, blur3)

    print('HPSNR')
    print('Floyd Steinberg got HPSNR\t= {}'.format(hpsnr_floyd))
    print('Jarvis got HPSNR\t= {}'.format(hpsnr_jarvis))
    print('Stucki got HPSNR\t= {}'.format(hpsnr_stucki))

    if show == 1:
        cv2.imshow('Original', gray)
        cv2.imshow('Filter 3X3', result3)
        cv2.imshow('Floyd Steinberg', gray1)
        cv2.imshow('Jarvis', gray2)
        cv2.imshow('Stucki', gray3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

single_run('Lenna',0)
single_run('boat',0)