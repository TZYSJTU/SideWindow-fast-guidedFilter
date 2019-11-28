import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
from SSIM_PIL import compare_ssim



def psnr(img1, img2):
   mse = torch.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * torch.log(PIXEL_MAX / torch.sqrt(mse))
   
# def psnr(img1, img2):
#    im1 = img1[0,0,2:-3,2:-3]
#    im2 = img2[0,0,2:-3,2:-3]
#    mse = torch.mean( (im1/255. - im2/255.) ** 2 )
#    if mse < 1.0e-10:
#       return 100
#    PIXEL_MAX = 1
#    return 20 * torch.log(PIXEL_MAX / torch.sqrt(mse))


class GuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-6):
        super(GuidedFilter, self).__init__()

        self.r = radius
        self.eps = eps

        self.pad = nn.ReplicationPad2d(radius)
        self.box  = nn.Conv2d(3, 3, kernel_size=2*radius+1, padding=0, dilation=1, bias=False, groups=3)
        self.box.weight.data[...] = 1.0/( 2*radius + 1)**2

        self.boxfilter = nn.Sequential(
            self.pad,
            self.box,
        )

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * 1 + 1 and w_x > 2 * 1 + 1

        # mean_x
        mean_x = self.boxfilter(x) 
        # mean_y
        mean_y = self.boxfilter(y) 
        # cov_xy
        cov_xy = self.boxfilter(x * y) - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) 
        mean_b = self.boxfilter(b) 

        res = mean_A * x + mean_b

        res = res.int()
        res = torch.clamp(res,0,255)
        res = res.float()

        return res


        
if __name__ == '__main__':

    s = GuidedFilter(1)
    
    x = Image.open('hr_ldr.jpg')
    x = np.array(x)
    x = np.transpose(x,(2,0,1))   
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0)

    y = Image.open('hr_hdr.jpg')
    y = np.array(y)
    y = np.transpose(y,(2,0,1))   
    y = torch.tensor(y, dtype=torch.float32)
    y = y.unsqueeze(0)
    print('input.shape = ', y.shape)

    res = s.forward(x,y)

    
    image1 = Image.open('hr_hdr.jpg')
    image2 = np.transpose(np.squeeze(res.detach().numpy()), (1,2,0))
    image2 = Image.fromarray(image2.astype(np.uint8))
    ssim = compare_ssim(image1, image2)
    print("ssim =",ssim)
    print("psnr =",psnr(res,y))

    res = torch.cat((y,res,x), dim=3)
    print(res.shape)
    res = np.transpose(np.squeeze(res.detach().numpy()), (1, 2, 0))
    res = res.astype(np.uint8)
    res = Image.fromarray(res)  
    res.show()