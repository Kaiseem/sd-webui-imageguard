import gradio as gr
import numpy as np
from modules import sd_models
from modules.ui import create_refresh_button
from imageguard.attacker import PGDAttacter
from tqdm import tqdm

def do_attack(m1,m2,m3,m4,m5, esb_atk_enable, image, text,  tgt_image, atksteps, epsilon, stepsize):
    model_list=[m1,m2,m3,m4,m5]
    if not esb_atk_enable:
        model_list=model_list[:1]
    model_list= [m for m in model_list if m != 'None']
    if len(model_list) == 0:
        return "Error: you must choose at least one model"
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

def on_ui_tabs():
    model_dropdowns = []
    with gr.Blocks(analytics_enabled=False) as attack_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                # input components

                with gr.Tabs():
                    with gr.TabItem(label='Choose Model and Image'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

                        text = gr.Textbox(
                            label="Guide Text",
                            info="Describe the main body of the given figure.",
                            lines=2,
                            value="A photo of a xx person.",
                        )

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
                    with gr.TabItem('PGD Settings'):
                        # pgd_params = gr.CheckboxGroup(["attack latent", "start noise", "vae sample"],
                        #                               value=["attack latent", "start noise", "vae sample"], label="PGD",
                        #                               elem_id="checkpoint_format")

                        atksteps = gr.Slider(label='Attack Step',minimum=0, maximum=200, value=100 )

                        epsilon = gr.Slider(
                            label='Epsilon',
                            minimum=0,
                            maximum=1,
                            value=0.05
                        )

                        stepsize = gr.Slider(
                            label='Step Size',
                            minimum=0,
                            maximum=1,
                            value=0.005
                        )

                    with gr.Tabs(elem_id="imageguard_model_checkbox"):
                        esb_atk_enable = gr.Checkbox(label="Ensemble Attack Enable")
                        with gr.TabItem('Attack Models'):
                            for i in range(0, 4):
                                model = gr.Dropdown(["None"] + sd_models.checkpoint_tiles(),
                                                    elem_id=f"model_attack_{i}", value="None",
                                                    label=f"Model {i + 1}")
                                # create_refresh_button(model, sd_models.list_models,
                                #                       lambda: {"choices": ["None"] + sd_models.checkpoint_tiles()},
                                #                       "refresh_checkpoint_Z")
                                model_dropdowns.append(model)


                    tgt_atk_enable = gr.Checkbox(label="Target Attack Enable")
                    tgt_image = gr.Image(
                            label='target attack image (TODO, not work now.)',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )


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

        model_attack.click(fn=do_attack,inputs=[*model_dropdowns,esb_atk_enable, image, text, tgt_image,atksteps,epsilon,stepsize],outputs=[adv_image])

    return [(attack_interface, "ImageGuard", "imageguard")]




