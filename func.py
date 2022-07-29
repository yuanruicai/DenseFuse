import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import pickle


EPSILON = 1e-10


def gaussian_pyramid(source, layers):
    source = source.copy()
    gp = [source]
    for i in range(layers - 1):
        source = cv2.pyrDown(source)
        gp.append(source)
    return gp


def laplacian_pyramid(source, layers):
    source = source.copy()
    gp = gaussian_pyramid(source, layers)
    lp = [gp[-1]]
    for i in range(layers - 1, 0, -1):
        ge = cv2.pyrUp(gp[i])
        print(gp[i - 1].shape, ge.shape)
        L = cv2.subtract(gp[i - 1], ge)
        lp.append(L)
    return lp


def reconstruct(lp):
    layers = len(lp)
    rec = lp[0]
    for i in range(1, layers):
        rec = cv2.pyrUp(rec)
        rec = cv2.add(rec, lp[i])
    return rec


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size=11, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=None, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padding = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1 * mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1 * mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class LossFunc(nn.Module):
    def __init__(self, lam=100, window_size=11, size_average=True, val_range=None):
        super(LossFunc, self).__init__()
        self.lam = lam
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.mse = 0
        self.ssim = 0

        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, x, y):
        self.mse = F.mse_loss(x, y)
        self.ssim = 1 - self._ssim(x, y)
        # print('mse:', MSE_loss.item(), 'ssim:', SSIM_loss.item())
        return self.mse + self.lam * self.ssim

    def _ssim(self, x, y):
        _, channel, _, _ = x.size()

        if channel == self.channel and self.window.dtype == x.dtype:
            window = self.window
            window = window.to(x.device)
        else:
            window = create_window(self.window_size, channel).to(x.device).type(x.dtype)
            self.window = window
            self.channel = channel
        return ssim(x, y, self.window_size, window, self.size_average)


def addition_fusion(*tensor):
    _sum = 0
    for t in tensor:
        _sum = _sum + t
    return _sum / len(tensor)


def activity_map(tensor, r=1):
    C = tensor.sum(dim=1, keepdim=True)
    P = nn.ReflectionPad2d(r)
    C = P(C)
    C_hat = F.avg_pool2d(input=C, kernel_size=2 * r + 1, stride=1)
    print('act', C_hat.size())
    return C_hat


def l1_norm_fusion(*tensor, mode='sum'):
    def _sum(maps, mode):
        s = 0
        for m in maps:
            if mode == 'softmax':
                s = s + torch.exp(m)
            if mode == 'sum':
                s = s + m
        return s + EPSILON

    shape = tensor[0].size()
    activity_maps = []
    weight_maps = []
    for i in tensor:
        activity_maps.append(activity_map(i, 1))
    den = _sum(activity_maps, mode)
    for i in activity_maps:
        if mode == 'softmax':
            weight_maps.append(torch.exp(i) / den)
        if mode == 'sum':
            weight_maps.append(i / den)
    tensor_f = 0
    for i, j in zip(weight_maps, tensor):
        weight = i.repeat(1, shape[1], 1, 1)
        tensor_f = tensor_f + weight * j
    return tensor_f


def Entropy(image):  # numpy gray
    h, w = image.shape
    tmp = 0.
    for k in range(256):
        p = np.sum(image == k) / (h*w)
        if p != 0:
            tmp = tmp - p * np.log2(p)
    return tmp


def save_object(filename="undistorted.pickle",obj=None):
    """后缀是.pickle"""
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


if __name__ == "__main__":
    path = "D:\\Download\\TNO_Image_Fusion_Dataset\\Athena_images\\lake"
    vis = cv2.imread(path + "\\VIS_lake_r.bmp", flags=cv2.IMREAD_GRAYSCALE)
    vis = cv2.resize(vis, (384, 288))
    LP = laplacian_pyramid(vis, 5)
    rec = reconstruct(LP)
    plt.figure()
    plt.imshow(np.hstack((vis, rec)), cmap='gray')
    plt.show()
