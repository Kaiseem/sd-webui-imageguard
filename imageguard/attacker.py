
import numpy as np
from typing import Tuple, List, Dict

from PIL import Image


from modules import shared

from modules import sd_models

from torch.cuda.amp import autocast as autocast
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
    def __init__(self, model_name, image,text) -> None:
        self.model_name = model_name
        self.load_model()
        self.image = preprocess_img(image, device_name).to(device_name)
        self.text=text

    def load_model(self):
        model_info = sd_models.checkpoints_list[self.model_name]
        self.model = sd_models.load_model(model_info)

    def attack(self):
        raise NotImplementedError()

class PGDAttacter(Attacker):
    def __init__(
        self,
        model,
        image,
        text,
        tgt_image,
        atksteps,
        epsilon,
        stepsize,
    ) -> None:
        super().__init__(model,image,text)
        self.tgt_attack=True if tgt_image is not None else False
        self.tgt_image = tgt_image
        self.atksteps = atksteps
        self.epsilon = epsilon
        self.step_size = stepsize
        self.step_sign = 1

    def attack(self,pbar,  return_no_norm_img=False):
        if self.tgt_attack:
            self.tgt_image = preprocess_img(self.tgt_image, device_name)
            self.tgt_image = F.interpolate(self.tgt_image, size=self.image.size()[2:], mode='bilinear')

        x = Variable(self.image.detach()).half()
        for i in range(self.atksteps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill(0)
            with autocast():
                latents = self.model.first_stage_model.encode(x).sample() * self.model.scale_factor  # N=4, C, 64, 64

                latents=latents.to(torch.float32)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    self.model.num_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                noisy_latents = self.model.q_sample(x_start=latents, t=timesteps, noise=noise)
                encoder_hidden_states = self.model.cond_stage_model([self.text])
                model_pred = self.model.apply_model(noisy_latents, timesteps, encoder_hidden_states)

            if self.model.parameterization == "eps":
                target = noise
            elif self.model.parameterization == "x0":
                target = latents
            elif self.model.parameterization == "v":
                target = self.model.get_v(latents, noise, timesteps)
            else:
                raise NotImplementedError(f"Paramterization {self.model.parameterization} not yet supported")

            self.model.zero_grad()
            with autocast():
                loss = F.mse_loss(model_pred, target, reduction="mean")
            loss.backward()

            x_adv = x.data + self.step_sign * self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, self.image - self.epsilon), self.image + self.epsilon)
            x_adv = torch.clamp(x_adv, -1, 1)
            x = Variable(x_adv)
            pbar.update(1)
        if return_no_norm_img:
            attacked_image=x.detach()
        else:
            attacked_image = Image.fromarray(to_rgb(x.detach()))
        del x, loss, x_adv, self.model
        return attacked_image

    def convert_to_rgb(self,x):
        return Image.fromarray(to_rgb(x))