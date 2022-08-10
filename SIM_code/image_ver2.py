from email.mime import image
import cv2  #OpenCVのインポート
import numpy as np #numpyをnpという名前でインポート
import random #randomをインポート
from matplotlib import pyplot as plt
import os



'''Make beads distribution image'''
def make_image(xypix, n_obj, circle_pix):
#400要素X600要素X3要素で全要素の値が0の3次元配列を生成しオブジェクトimgに代入
    img=np.zeros((xypix[0], xypix[1]), np.uint8) 

    for i in range(n_obj): #以下のインデント行を10回繰り返す
        x=int(random.uniform(circle_pix*2, xypix[0]-2*circle_pix)) #10以上590以下の乱数(浮動小数点型)を発生し、intで整数に変換、xとする
        y=int(random.uniform(circle_pix*2, xypix[1]-2*circle_pix)) #10以上390以下の乱数(浮動小数点型)を発生し、intで整数に変換、yとする　
        cv2.circle(img, (x, y), int(circle_pix), (1, 1, 1), -1) #中心がx,y半径が10の水色の塗りつぶした円を描画
    
    return img+10

'''Make structured light image'''   
def make_structure(xypix, k, theta, phase_ill):
    C=1.0
    img_str=[[0]*xypix[0] for i in range(xypix[1])]
    kill=[k*np.cos(theta),k*np.sin(theta)]
    for i in range(xypix[0]):
        for j in range(xypix[1]):
            img_str[i][j]=1+C*np.cos(kill[0]*i+kill[1]*j+phase_ill)
        
    return img_str

'''Make convolution image'''
def convolve2d(image, kernel, boundary='edge'):
    if boundary is not None:
        pad_image=np.pad(image, ((int(kernel.shape[0] / 2),), (int(kernel.shape[1] / 2),)), boundary)
    else:
        pad_image=image
    
    shape = (pad_image.shape[0] - kernel.shape[0] + 1, pad_image.shape[1] - kernel.shape[1] + 1) + kernel.shape
    strides = pad_image.strides * 2
    strided_image = np.lib.stride_tricks.as_strided(pad_image, shape, strides)
    return np.einsum('kl,ijkl->ij', kernel, strided_image)


def main():
    dirnames= ["results/fig/","results/csv/"]
    for f_name in dirnames:
        os.makedirs(f_name, exist_ok=True)

    Light_lambda=633*10**(-1) #Structured light wavelength
    k=2*np.pi/Light_lambda #Wave number
    theta=0*np.pi #Incident angle
    phase_ill=np.pi*2 #Phase delay

    #Read file of PSF
    Wavelength_psf=633 #nm
    kernel_data_na055 =np.genfromtxt("Intensity_Na0.55_wavelength"+str(Wavelength_psf)+"nm.csv", delimiter=",", skip_header=0, dtype='float', filling_values=0)
    kernel_data_na1 =np.genfromtxt("Intensity_Na1_wavelength"+str(Wavelength_psf)+"nm.csv", delimiter=",", skip_header=0, dtype='float', filling_values=0)
    
    #Make structure light image
    xypix=[1000,1000] #Pixcel number of image
    img_str=np.abs(make_structure(xypix,k,theta,phase_ill)) #Make absolute image
    
    #Parameters to make beads distribution image

    Mode_beads=0 #0: Make image, 1: Read image

    n_obj=100 #Number of beads
    circle_r=500 #Radius of beads
    circle_pix=circle_r/10 #Size correction

    #Make or read beads distribution image
    if Mode_beads==0:
        img=make_image(xypix, n_obj, circle_pix) #Make image
    elif Mode_beads==1:
        img=np.genfromtxt("random_circle_radius"+circle_r+"nm_pix1000_img.csv", delimiter=",", skip_header=0, dtype='float', filling_values=0)
    
    #Make convolution images
    img_cnv_na055=convolve2d(img,kernel_data_na055)
    img_cnv_na1=convolve2d(img,kernel_data_na1)
    img_str_cnv_psf_na055=convolve2d(img_str,kernel_data_na055)
    img_str_cnv_psf_na1=convolve2d(img_str,kernel_data_na1)   
    img_inter_norm=img/np.max(img)*img_str/np.max(img_str) #Bead distribution image x structured light image
    #img_inter=img_str/np.max(img_str)+img/np.max(img)
    #img_inter_norm=img_inter/np.max(img_inter)
    img_inter_cnv_psf_na055=convolve2d(img_inter_norm,kernel_data_na055)
    img_inter_cnv_psf_na1=convolve2d(img_inter_norm,kernel_data_na1)

    file_name={'img_str':img_str,'img':img,'img_inter_norm':img_inter_norm,'img_cnv_na055':img_cnv_na055,'img_cnv_na1':img_cnv_na1,'img_str_cnv_psf_na055':img_str_cnv_psf_na055,'img_str_cnv_psf_na1':img_str_cnv_psf_na1,'img_inter_cnv_psf_na055':img_inter_cnv_psf_na055,'img_inter_cnv_psf_na1':img_inter_cnv_psf_na1}
    for key,value in file_name.items():
        plt.imshow(value, cmap = 'gray')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.savefig(dirnames[0]+"random_circle_radius"+str(circle_r)+"nm_pix"+str(xypix[0])+"_"+key+".png")
        np.savetxt(dirnames[1]+"random_circle_radius"+str(circle_r)+"nm_pix"+str(xypix[0])+"_"+key+".csv", img, delimiter=",")
        plt.show()
        plt.clf()    

main()
