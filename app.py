import random

import gradio as gr
from PIL.Image import Image
from loadimg import load_img

from daggr import GradioNode, Graph, FnNode

glm_image = GradioNode(
    "hf-applications/Z-Image-Turbo",
    api_name="/generate_image",
    inputs={
        "prompt": gr.Textbox(  # An input node is created for the prompt
            label="Prompt",
            value="A cheetah in the grassy savanna.",
            lines=3,
        ),
        "height": 1024,  # Fixed value (does not appear in the canvas)
        "width": 1024,  # Fixed value (does not appear in the canvas)
        "seed": random.random,  # Functions are rerun every time the workflow is run (not shown in the canvas)
    },
    outputs={
        "image": gr.Image(
            label="Image"  # Display original image
        ),
    },
)

background_remover = GradioNode(
    "hf-applications/background-removal",
    api_name="/image",
    inputs={
        "image": glm_image.image,
    },
    postprocess=lambda _, final: final,
    outputs={
        "image": gr.Image(label="Final Image"),  # Display only final image
    },
)


def crop_alpha(image: Image) -> Image:
    """crops image keep only the RGB channels"""

    # convert from str to PIL Image
    image = load_img(image).convert("RGBA")
    bbox = image.getbbox(alpha_only=True)
    image = image.crop(bbox)
    # store as str and pass as path
    return load_img(image, output_type="str")


cropper = FnNode(
    fn=crop_alpha,
    inputs={
        "image": background_remover.image,
    },
    outputs={
        "image": gr.Image(label="crops image to fit by removing alpha border"),
    },
)


graph = Graph(
    name="Transparent Background Image Generator",
    nodes=[glm_image, background_remover, cropper],
)

graph.launch()
