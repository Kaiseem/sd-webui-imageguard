import gradio as gr
import numpy as np
from modules import sd_models
from modules.ui import create_refresh_button
from imageguard.attacker import PGDAttacter
from tqdm import tqdm
import os
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from glob import glob

from typing import Dict, Callable, NamedTuple
from pathlib import Path
import re



def do_attack(m1, m2, m3, m4, m5, esb_atk_enable, image, batch_input_glob, batch_input_recursive, batch_output_dir,
              text, tgt_image, atksteps, epsilon, stepsize):
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()
    model_list=[m1,m2,m3,m4,m5]
    if not esb_atk_enable:
        model_list = model_list[:1]
    model_list = [m for m in model_list if m != 'None']
    if len(model_list) == 0:
        return "Error: you must choose at least one model"

    if batch_input_glob == '':
        if image is None:
            return "Error: you must choose an image"
        pbar = tqdm(total=atksteps*len(model_list))
        if len(model_list)==1:
            attacker=PGDAttacter(model_list[0], image,text, tgt_image, atksteps, epsilon, stepsize)
            attacked_image = attacker.attack(pbar)
        else:
            attacked_image=0
            for i, m in enumerate(model_list):
                attacker = PGDAttacter(m, image, text, tgt_image, atksteps, epsilon, stepsize)
                temp_image = attacker.attack(pbar, return_no_norm_img=True)
                attacked_image += temp_image/len(model_list)
                if i ==len(model_list)-1:
                    attacked_image=attacker.convert_to_rgb(attacked_image)
        pbar.close()
        return attacked_image
    else:
        # if there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # check the input directory path
        if not os.path.isdir(base_dir):
            return 'Error: Input path is not a directory'

        # this line is moved here because some reason
        # PIL.Image.registered_extensions() returns only PNG if you call too early
        supported_extensions = [
            e
            for e, f in Image.registered_extensions().items()
            if f in Image.OPEN
        ]

        paths = [
            Path(p)
            for p in glob(batch_input_glob, recursive=batch_input_recursive)
            if '.' + p.split('.').pop().lower() in supported_extensions
        ]

        print(f'found {len(paths)} image(s)')

        for path in paths:
            try:
                image = Image.open(path)
            except UnidentifiedImageError:
                # just in case, user has mysterious file...
                print(f'${path} is not supported image type')
                continue

            # guess the output path
            base_dir_last = Path(base_dir).parts[-1]
            base_dir_last_idx = path.parts.index(base_dir_last)
            output_dir = Path(batch_output_dir) if batch_output_dir else Path(base_dir+'_atk')
            output_dir = output_dir.joinpath(*path.parts[base_dir_last_idx + 1:]).parent
            output_dir.mkdir(0o777, True, True)
            output_path=output_dir.joinpath(path.parts[-1])
            print(f'Processing {path.parts[-1]}')
            pbar = tqdm(total=atksteps * len(model_list))
            if len(model_list) == 1:
                attacker = PGDAttacter(model_list[0], image, text, tgt_image, atksteps, epsilon, stepsize)
                attacked_image = attacker.attack(pbar)
            else:
                attacked_image = 0
                for i, m in enumerate(model_list):
                    attacker = PGDAttacter(m, image, text, tgt_image, atksteps, epsilon, stepsize)
                    temp_image = attacker.attack(pbar, return_no_norm_img=True)
                    attacked_image += temp_image / len(model_list)
                    if i == len(model_list) - 1:
                        attacked_image = attacker.convert_to_rgb(attacked_image)
            pbar.close()
            attacked_image.save(output_path)

        print('All Done')
        return None


def on_ui_tabs():
    model_dropdowns = []
    with gr.Blocks(analytics_enabled=False) as attack_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                # input components

                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

                    with gr.TabItem(label='Batch from directory'):
                        batch_input_glob = gr.Textbox(
                            label='Input directory',
                            placeholder='/path/to/images or /path/to/images/**/*'
                        )
                        batch_input_recursive = gr.Checkbox(
                            label='Use recursive with glob pattern'
                        )

                        batch_output_dir = gr.Textbox(
                            label='Output directory',
                            placeholder='Leave blank to save images to the same path.'
                        )

                    text = gr.Textbox(
                            label="Input prompt",
                            info="Describe the main body of the given figure.",
                            lines=2,
                            value="A photo of a xx person.",)

                    model = gr.Dropdown(["None"] + sd_models.checkpoint_tiles(),
                                            elem_id="model_attack_selected", value="None",
                                            label="Model")
                    create_refresh_button(model, sd_models.list_models,
                                              lambda: {"choices": ["None"] + sd_models.checkpoint_tiles()},
                                              "refresh_checkpoint_Z")
                    model_dropdowns.append(model)

                    model_attack = gr.Button(
                        elem_id="imageguard_model_attack",
                        value='Protect',
                        variant='primary'
                    )

                with gr.Tabs(elem_id="imageguard_pgd_checkbox"):
                    with gr.TabItem('PGD Attack Parameters'):
                        atksteps = gr.Slider(label='Attack iterations',minimum=0, maximum=200, value=100 )
                        epsilon = gr.Slider(label='Epsilon',minimum=0,maximum=1,value=0.05)
                        stepsize = gr.Slider(label='Step size',minimum=0,maximum=1,value=0.005)

                with gr.Tabs(elem_id="imageguard_model_checkbox"):
                    with gr.TabItem('Ensemble Attack'):
                        esb_atk_enable = gr.Checkbox(label="Ensemble Attack Enable")
                        for i in range(0, 4):
                            model = gr.Dropdown(["None"] + sd_models.checkpoint_tiles(),
                                                    elem_id=f"model_attack_{i}", value="None",
                                                    label=f"Model {i + 1}")
                                # create_refresh_button(model, sd_models.list_models,
                                #                       lambda: {"choices": ["None"] + sd_models.checkpoint_tiles()},
                                #                       "refresh_checkpoint_Z")
                            model_dropdowns.append(model)

                    with gr.TabItem('Target Attack'):
                        tgt_atk_enable = gr.Checkbox(label="Target Attack Enable")
                        tgt_image = gr.Image(
                                label='target attack image (TODO, not work now.)',
                                source='upload',
                                interactive=True,
                                type="pil")


            # output components
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.Row():
                        with gr.TabItem(label='Output Image'):
                            adv_image = gr.Image(
                                label='atk_img',
                                source='upload',
                                interactive=False,
                                type="pil"
                            )

        model_attack.click(fn=do_attack, inputs=[*model_dropdowns,
                                                 esb_atk_enable,

                                                 # single process
                                                 image,

                                                 # batch process
                                                 batch_input_glob,
                                                 batch_input_recursive,
                                                 batch_output_dir,

                                                 # options
                                                 text,
                                                 tgt_image,

                                                 # PGD params
                                                 atksteps,
                                                 epsilon,
                                                 stepsize
                                                 ],
                           outputs=[adv_image])

    return [(attack_interface, "ImageGuard", "imageguard")]




