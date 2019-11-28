import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import cv2
import numpy as np


class sFastGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-4, magnification=8):
        super(sFastGuidedFilter, self).__init__()

        self.r = radius
        self.eps = eps
        self.magnification = magnification

        self.pad = nn.ReplicationPad2d(radius)
        self.box  = nn.Conv2d(3, 3, kernel_size=2*radius+1, padding=0, dilation=1, bias=False, groups=3)
        self.box.weight.data[...] = 1.0/( 2*radius + 1)**2
        
        self.boxfilter = nn.Sequential(
            self.pad,
            self.box,
        )


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1
        
        ## mean_x
        mean_x = self.boxfilter(lr_x) 
        ## mean_y
        mean_y = self.boxfilter(lr_y) 
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        # mean_A = self.boxfilter(A) 
        # mean_b = self.boxfilter(b) 

        # lr_y = mean_A * lr_x + mean_b
        

        # A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        # b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        print("A.shpae = ", A.size())
        _, c, h, w = A.size()
        d = torch.zeros(b, 8, h, w, dtype=torch.float32)
        
        filter = torch.ones(1, 1, 2*self.r+1, 2*self.r+1)
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

        # res = lr_x.clone()
        # print(res.size())
        mask_min = torch.zeros(1,c,h,w)
        for ch in range(3):
            A_ = A[:,ch].view(1,1,h,w)
            b_ = b[:,ch].view(1,1,h,w)
            im_ch = lr_x[0, ch, ::].clone().view(1, 1, h, w)

            d[:, 0, ::] = F.conv2d(input=A_, weight=L, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=L, padding=(self.r, self.r)) - im_ch
            d[:, 1, ::] = F.conv2d(input=A_, weight=R, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=R, padding=(self.r, self.r)) - im_ch
            d[:, 2, ::] = F.conv2d(input=A_, weight=U, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=U, padding=(self.r, self.r)) - im_ch
            d[:, 3, ::] = F.conv2d(input=A_, weight=D, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=D, padding=(self.r, self.r)) - im_ch
            d[:, 4, ::] = F.conv2d(input=A_, weight=NW, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=NW, padding=(self.r, self.r)) - im_ch
            d[:, 5, ::] = F.conv2d(input=A_, weight=NE, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=NE, padding=(self.r, self.r)) - im_ch
            d[:, 6, ::] = F.conv2d(input=A_, weight=SW, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=SW, padding=(self.r, self.r)) - im_ch
            d[:, 7, ::] = F.conv2d(input=A_, weight=SE, padding=(self.r, self.r)) * im_ch + F.conv2d(input=b_, weight=SE, padding=(self.r, self.r)) - im_ch

            d_abs = torch.abs(d)
            #print('im_ch', im_ch)
            #print('dm = ', d_abs.shape, d_abs)
            mask_min[b,c] = torch.argmin(d_abs, dim=1, keepdim=True).squeeze()
            #print('mask min = ', mask_min.shape, mask_min)
            dm = torch.gather(input=d, dim=1, index=mask_min)
            im_ch = dm + im_ch

            res[:, ch, ::] = im_ch
        res = res.int()
        res = torch.clamp(res,0,255)
        res = res.float()
        return res

if __name__ == '__main__':

    s = sFastGuidedFilter(1)

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

    res = hr_guide
    _,_,h,w = lr_guide.size()
    for iteration in range (3):
        res = s.forward(lr_guide, lr_output ,res)
        lr_guide = F.interpolate(res, (h, w), mode='bilinear', align_corners=True)

    res = torch.cat((hr_label,res,hr_guide), dim=3)
    print("res =", res.shape)
    res = np.transpose(np.squeeze(res.data.numpy()), (1, 2, 0))
    res = res.astype(np.uint8)
    res = Image.fromarray(res)  
    res.show()
