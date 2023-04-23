from PIL import Image, ImageFile
from modules import script_callbacks
from imageguard.ui import on_ui_tabs

Image.init()
ImageFile.LOAD_TRUNCATED_IMAGES = True
script_callbacks.on_ui_tabs(on_ui_tabs)
