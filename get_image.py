import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image, ImageFilter 

if __name__ == '__main__':
    hr_hdr = Image.open('hr_hdr.jpg')
    hr_ldr = Image.open('hr_ldr.jpg')
    # hr_noise = hr.filter(ImageFilter.GaussianBlur(radius=5)) 
    # hr_noise.save('D:/lib\论文/子窗口滤波/hr_noise.jpg')
    # hr_noise = Image.open('hr_ldr.jpg')
    h, w = int(hr_hdr.size[0]/8), int(hr_hdr.size[1]/8)
    
    lr_hdr = hr_hdr.resize((h, w))
    lr_ldr = hr_ldr.resize((h, w))
    lr_hdr.save('D:/lib\论文/子窗口滤波/lr_hdr.jpg')
    lr_ldr.save('D:/lib\论文/子窗口滤波/lr_ldr.jpg')

    lr_hdr = np.array(lr_hdr)
    lr_ldr = np.array(lr_ldr)

    img = np.concatenate((lr_hdr, lr_ldr), axis=1)
    img = Image.fromarray(img)
    img.show()

 

