import torch
from torch import nn
from torch.nn import functional as F
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

class FastGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-2):
        super(FastGuidedFilter, self).__init__()

        self.r = radius
        self.eps = eps
        self.pad = nn.ReplicationPad2d(radius)
        self.box  = nn.Conv2d(3, 3, kernel_size=2*radius+1, padding=0, dilation=1, bias=False, groups=3)
        self.box.weight.data[...] = 1.0/( 2*radius + 1)**2

        self.boxfilter = nn.Sequential(
            self.pad,
            self.box,
        )

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        ## mean_x
        mean_x = self.boxfilter(x_lr)
        ## mean_y
        mean_y = self.boxfilter(y_lr)
        ## cov_xy
        cov_xy = self.boxfilter(x_lr * y_lr) - mean_x * mean_y
        ## var_x
        var_x  = self.boxfilter(x_lr * x_lr) - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) 
        mean_b = self.boxfilter(b) 

        mean_A = F.interpolate(mean_A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(mean_b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        res = mean_A * x_hr + mean_b
        res = torch.clamp(res,0,255)
        res = res.float()

        return res


if __name__ == '__main__':

    s = FastGuidedFilter(1)

    hr_label = Image.open('hr_hdr.jpg')
    hr_label = np.array(hr_label)
    hr_label = np.transpose(hr_label,(2,0,1))   
    hr_label = torch.tensor(hr_label, dtype=torch.float32)
    hr_label = hr_label.unsqueeze(0)
    print('hr_label = ', hr_label.shape)

    hr_guide = Image.open('hr_ldr.jpg')
    hr_guide = np.array(hr_guide)
    hr_guide = np.transpose(hr_guide,(2,0,1))   
    hr_guide = torch.tensor(hr_guide, dtype=torch.float32)
    hr_guide = hr_guide.unsqueeze(0)

    lr_output = Image.open('lr_hdr.jpg')
    lr_output = np.array(lr_output)
    lr_output = np.transpose(lr_output,(2,0,1))   
    lr_output = torch.tensor(lr_output, dtype=torch.float32)
    lr_output = lr_output.unsqueeze(0)

    lr_guide = Image.open('lr_ldr.jpg')
    lr_guide = np.array(lr_guide)
    lr_guide = np.transpose(lr_guide,(2,0,1))   
    lr_guide = torch.tensor(lr_guide, dtype=torch.float32)
    lr_guide = lr_guide.unsqueeze(0)

    res = s.forward(lr_guide, lr_output , hr_guide)

    image1 = Image.open('hr_hdr.jpg')
    image2 = np.transpose(np.squeeze(res.detach().numpy()), (1,2,0))
    image2 = Image.fromarray(image2.astype(np.uint8))
    ssim = compare_ssim(image1, image2)
    print("ssim =",ssim)
    print("psnr =",psnr(res,hr_label))


    res = torch.cat((hr_label,res,hr_guide), dim=3)
    print("res =", res.shape)
    res = np.transpose(np.squeeze(res.data.numpy()), (1, 2, 0))
    res = res.astype(np.uint8)
    res = Image.fromarray(res)  
    res.show()

 
