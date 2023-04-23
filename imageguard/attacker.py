
import numpy as np
from typing import Tuple, List, Dict

from PIL import Image


from modules import shared

from modules import sd_models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from imageguard.utils import zero_gradients, preprocess_img, to_rgb

use_cpu = ('all' in shared.cmd_opts.use_cpu)

if use_cpu:
    device_name = 'cpu'
else:
    device_name = 'cuda:0'
    if shared.cmd_opts.device_id is not None:
        try:
            device_name = f'cuda:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')

class Attacker:
    def __init__(self, model_name, image) -> None:
        print('DEBUG:',device_name)
        self.model_name = model_name
        self.load_model()
        self.image = preprocess_img(image, device_name).to(device_name)

    def load_model(self):
        model_info = sd_models.checkpoints_list[self.model_name]
        self.model = sd_models.load_model(model_info)
        self.model.to(device_name)

    def attack(self):
        raise NotImplementedError()

class PGDAttacter(Attacker):
    def __init__(
        self,
        model,
        image,
        tgt_image,
        atksteps,
        epsilon,
        stepsize,
        pgd_params
    ) -> None:
        super().__init__(model,image)
        self.tgt_attack=True if tgt_image is not None else False
        self.tgt_image = tgt_image
        self.atksteps = atksteps
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.atk_latent = True if "attack latent" in pgd_params else False
        self.start_noise = True if "start noise" in pgd_params else False
        self.vae_sample = True if "vae sample" in pgd_params else False

    def attack(self):
        if self.tgt_attack:
            self.tgt_image = preprocess_img(self.tgt_image, device_name)
            self.tgt_image = F.interpolate(self.tgt_image, size=self.image.size()[2:], mode='bilinear')

        loss_fn = nn.MSELoss()
        epsilon = self.epsilon / 255 * 2
        step_size = self.stepsize / 255 * 2
        x = Variable(self.image.detach()) if not self.start_noise else Variable(
            self.image.detach() + torch.zeros_like(self.image).uniform_(-epsilon, epsilon))

        if not self.tgt_attack:
            step_sign = 1
            if self.atk_latent:
                with torch.no_grad():
                    label = self.model.first_stage_model.encode(self.image)
            else:
                label = self.image.detach()
        else:
            step_sign = -1
            if self.atk_latent:
                with torch.no_grad():
                    label = self.model.first_stage_model.encode(self.tgt_image)
            else:
                label = self.tgt_image.detach()

        for i in range(self.atksteps):
            print('step ' + str(i))
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill(0)
            if self.vae_sample:
                z = self.model.first_stage_model.encode(x).sample().half()
            else:
                z = self.model.first_stage_model.encode(x).mode().half()

            if self.atk_latent:
                loss = loss_fn(z, label.mode().half())
            else:
                out = self.model.first_stage_model.decode(z)

                loss = loss_fn(out, label)

            self.model.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, self.image - epsilon), self.image + epsilon)
            x_adv = torch.clamp(x_adv, -1, 1)
            x = Variable(x_adv)

        with torch.no_grad():
            z = self.model.first_stage_model.encode(x).mode().half()
            recon_x = self.model.first_stage_model.decode(z)

        attacked_image = Image.fromarray(to_rgb(x.detach()))
        reconstructed_image = Image.fromarray(to_rgb(recon_x))
        del x, loss, x_adv, label
        return attacked_image, reconstructed_image