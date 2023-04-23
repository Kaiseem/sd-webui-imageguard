import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()

def preprocess_img(pil_img, device_name):
    pil_img=torch.from_numpy(np.array(pil_img).transpose((2,0,1))).unsqueeze(0).to(device_name).half()
    pil_img= pil_img/127.5-1
    return pil_img

def to_rgb(tensor):
    tensor = torch.clip(tensor,-1,1)
    tensor = tensor.detach().cpu().numpy()[0]*0.5+0.5
    img = np.clip(tensor.transpose((1, 2, 0)) * 255, 0, 255).astype(np.uint8)
    return img