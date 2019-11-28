import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
from SSIM_PIL import compare_ssim

# def psnr(img1, img2):
#    im1 = img1[0,0,2:-3,2:-3]
#    im2 = img2[0,0,2:-3,2:-3]
#    mse = torch.mean( (im1/255. - im2/255.) ** 2 )
#    if mse < 1.0e-10:
#       return 100
#    PIXEL_MAX = 1
#    return 20 * torch.log(PIXEL_MAX / torch.sqrt(mse))
def psnr(img1, img2):
   mse = torch.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * torch.log(PIXEL_MAX / torch.sqrt(mse))

class sGuidedFilter(nn.Module):
    def __init__(self, radius, eps=1e-6):
        super(sGuidedFilter, self).__init__()

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
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # mean_x
        mean_x = self.boxfilter(x) 
        # mean_y
        mean_y = self.boxfilter(y) 
        # cov_xy
        cov_xy = self.boxfilter(x * y)  - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        print("A.shape =", A.size())
        _, _, h, w = A.size()
        # d = torch.zeros(1, 8, h, w, dtype=torch.float32)
        d = torch.zeros(1, 8, h, w, dtype=torch.float32)


        filter = torch.ones(1, 1, 2*self.r+1, 2*self.r+1)

        FULL = filter.clone() / ((self.r + 1) ** 2)
        L, R, U, D = [filter.clone() for _ in range(4)]
        L[:, :, :, self.r + 1:] = 0
        R[:, :, :, 0: self.r] = 0
        U[:, :, self.r + 1:, :] = 0
        D[:, :, 0: self.r, :] = 0

        NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

        L, R, U, D = L / ((self.r + 1) * (self.r*2 +1)), R / ((self.r + 1) * (self.r*2 +1)), \
                    U / ((self.r + 1) * (self.r*2 +1)), D / ((self.r + 1) * (self.r*2 +1))

        NW[:, :, :, self.r + 1:] = 0
        NE[:, :, :, 0: self.r] = 0
        SW[:, :, :, self.r + 1:] = 0
        SE[:, :, :, 0: self.r] = 0

        NW, NE, SW, SE = NW / ((self.r + 1) ** 2), NE / ((self.r + 1) ** 2), \
                        SW / ((self.r + 1) ** 2), SE / ((self.r + 1) ** 2)

        res = x.clone()

        for ch in range(3):

            A_ = A[:,ch].view(1,1,h,w)
            A_ = self.pad(A_)
            b_ = b[:,ch].view(1,1,h,w)
            b_ = self.pad(b_)

            im_ch = res[0, ch].clone().view(1, 1, h, w)

            d[:, 0, ::] = F.conv2d(input=A_, weight=L, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=L, padding=(0, 0)) - im_ch
            d[:, 1, ::] = F.conv2d(input=A_, weight=R, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=R, padding=(0, 0)) - im_ch
            d[:, 2, ::] = F.conv2d(input=A_, weight=U, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=U, padding=(0, 0)) - im_ch
            d[:, 3, ::] = F.conv2d(input=A_, weight=D, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=D, padding=(0, 0)) - im_ch
            d[:, 4, ::] = F.conv2d(input=A_, weight=NW, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=NW, padding=(0, 0)) - im_ch
            d[:, 5, ::] = F.conv2d(input=A_, weight=NE, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=NE, padding=(0, 0)) - im_ch
            d[:, 6, ::] = F.conv2d(input=A_, weight=SW, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=SW, padding=(0, 0)) - im_ch
            d[:, 7, ::] = F.conv2d(input=A_, weight=SE, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=SE, padding=(0, 0)) - im_ch
            # d[:, 8, ::] = F.conv2d(input=A_, weight=FULL, padding=(0, 0)) * im_ch + F.conv2d(input=b_, weight=FULL, padding=(0, 0)) - im_ch 

            d_abs = torch.abs(d)
            #print('im_ch', im_ch)
            #print('dm = ', d_abs.shape, d_abs)
            mask_min = torch.argmin(d_abs, dim=1, keepdim=True)
            #print('mask min = ', mask_min.shape, mask_min)
            dm = torch.gather(input=d, dim=1, index=mask_min)
            im_ch = dm + im_ch
            # print(im_ch.shape)
            res[0, ch,::] = im_ch[0,0,::]
            # print(res.shape)

        res = res.int()
        res = torch.clamp(res,0,255)
        res = res.float()

        return res

if __name__ == '__main__':

    s = sGuidedFilter(1)

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
    
    print('input.shape =', y.shape)

  
    res = s.forward(x,y)

    
    
    image1 = Image.open('hr_hdr.jpg')
    image2 = np.transpose(np.squeeze(res.detach().numpy()), (1,2,0))
    image2 = Image.fromarray(image2.astype(np.uint8))
    ssim = compare_ssim(image1, image2)
    print("ssim =",ssim)
    print("psnr =",psnr(res,y))

    res = torch.cat((y,res,x), dim=3)
    print("result.shape =", res.shape)
    res = np.transpose(np.squeeze(res.numpy()), (1, 2, 0))
    res = res.astype(np.uint8)
    res = Image.fromarray(res)  # numpy to image
    res.show()

    

