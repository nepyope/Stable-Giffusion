import os
os.environ["JAX_PLATFORMS"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import jax
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline    
import torch
from PIL import Image
model_path = 'sd-model'
prompt = 'Traveling on thai traditional wooden boat among limestone rocky islands. thailand tropical landscape, krabi province'
# load model and scheduler
pipe = StableDiffusionPipeline.from_pretrained('sd-model',safety_checker=None,from_flax=True).to("cuda")
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('sd-model',safety_checker=None,from_flax=True).to("cuda")

img = pipe(prompt=f'FRAME0 {prompt}', num_inference_steps=20, width=512, height=256).images[0]
img.save(f'frames/FRAME0.jpg')
for i in range(152):
    img = pipeline(prompt=f'FRAME{i+1} {prompt}',image=img, num_inference_steps=20).images[0]
    #save the image
    img.save(f'frames/FRAME{i+1}.jpg')
