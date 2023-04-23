import gradio as gr
from modules import sd_models
from modules.ui import create_refresh_button
from imageguard.attacker import PGDAttacter

def do_attack(model, image, tgt_image, atksteps, epsilon, stepsize, pgd_params):
    if model == "":
        return "Error: you must choose a model"
    if image is None:
        return "Error: you must choose an image"

    attacker=PGDAttacter(model, image, tgt_image, atksteps, epsilon, stepsize, pgd_params)
    attacked_image, reconstructed_image = attacker.attack()
    return attacked_image,reconstructed_image

def on_ui_tabs():
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

                    with gr.Row():
                        model_name = gr.Dropdown(sd_models.checkpoint_tiles(),
                                                 elem_id="model_converter_model_name",
                                                 label="Model")
                        create_refresh_button(model_name, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()},
                                              "refresh_checkpoint_Z")

                    model_attack = gr.Button(
                        elem_id="imageguard_model_attack",
                        value='Attack',
                        variant='primary'
                    )

                with gr.Tabs(elem_id="imageguard_checkbox"):
                    with gr.TabItem('Settings'):
                        pgd_params = gr.CheckboxGroup(["attack latent", "start noise", "vae sample"],
                                                      value=["attack latent", "start noise", "vae sample"], label="PGD",
                                                      elem_id="checkpoint_format")

                        atksteps = gr.Slider(label='Attack Step',minimum=0, maximum=200, value=20 )

                        epsilon = gr.Slider(
                            label='Epsilon',
                            minimum=0,
                            maximum=255,
                            value=8
                        )

                        stepsize = gr.Slider(
                            label='Step Size',
                            minimum=0,
                            maximum=255,
                            value=2
                        )

                    tgt_atk_enable = gr.Checkbox(label="Target Attack Enable")
                    tgt_image = gr.Image(
                            label='target attack image',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

            # output components
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.Row():
                        with gr.TabItem(label='Attacked image'):
                            adv_image = gr.Image(
                                label='atk_img',
                                source='upload',
                                interactive=False,
                                type="pil"
                            )
                with gr.Tabs():
                    with gr.Row():
                        with gr.TabItem(label='Reconstructed image'):
                            rec_image = gr.Image(
                                label='recon_img',
                                source='upload',
                                interactive=False,
                                type="pil"
                            )

        model_attack.click(fn=do_attack,inputs=[model_name, image,tgt_image,atksteps,epsilon,stepsize,pgd_params],outputs=[adv_image, rec_image])

    return [(attack_interface, "ImageGuard", "imageguard")]




