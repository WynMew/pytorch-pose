from __future__ import print_function, absolute_import
import os
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
from pose.progress.bar import Bar as Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray.copy())
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def im_to_torch(img):
    img = img[..., ::-1] #RGB
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

idx = [1, 2, 3, 4, 5, 6, 11, 12, 15, 16]

arch = 'hg'
stacks = 2
blocks = 1
num_classes = 16
resume = 'checkpoint/mpii/hg_s2_b1/model_best.pth.tar'
torch.cuda.set_device(0)
model = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=num_classes)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()


vc = cv2.VideoCapture(1)
vc.set(3,256)
vc.set(4,256)
while(1):
    if vc.isOpened():
        rval, frame =vc.read()
        break
    else:
        rval = False

print('camera ready')
frame = cv2.resize(frame,(256,256))
image = Image.fromarray(frame)
b, g, r = image.split()
image = Image.merge('RGB',(r,g,b))

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(image)
'''
videoname = 'PoseOut.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(videoname, fourcc, 20, (1280, 256))
'''
def updatefig(*args):
    rval, frame = vc.read()
    frame = cv2.resize(frame,(256,256))# BGR
    img = im_to_torch(frame)
    inp = color_normalize(img, (0.4404, 0.4440, 0.4327), (0.2458, 0.2410, 0.2468))
    input_var = torch.autograd.Variable(inp.cuda())
    output = model(input_var.unsqueeze(0))
    score_map = output[-1].data.cpu()
    pred_batch_img = batch_with_heatmap(inp.unsqueeze(0), score_map)
    ax.clear()
    ax.imshow(pred_batch_img)
#    oframe = np.array(image)
#    oframe = oframe[:, :, ::-1].copy()
#    outvideo.write(oframe)
    return ax

ani = animation.FuncAnimation(fig, updatefig, interval=1)
plt.show()


'''
transforms.Normalize((0.4404, 0.4440, 0.4327), (0.2458, 0.2410, 0.2468))])
'''
