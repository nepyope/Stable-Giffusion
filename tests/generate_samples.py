import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate samples from a the pretrained model.")

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='The quick brown fox jumps over the lazy dog.',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    dtype = jnp.bfloat16

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        "sd-test",
        revision="bf16",
        dtype=dtype,
        safety_checker=None,
    )

    prompt = args.prompt
    prompt = [prompt] * jax.device_count()
    prompt_ids = pipeline.prepare_inputs(prompt)
    print(prompt_ids.shape)
    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)
    print(prompt_ids.shape)
    def create_key(seed=0):
        return jax.random.PRNGKey(seed)

    rng = create_key(0)
    rng = jax.random.split(rng, jax.device_count())

    images = pipeline(prompt_ids, p_params, rng, jit=True,width=args.width,height=args.height)[0]

    # Define the local destination folder
    local_folder = "results"

    # Iterate over the images and save them to the local folder
    for i, image in enumerate(images):
        image = image[0]
        image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image)
        img.save(f"{local_folder}/image_{i}.png")

if __name__ == "__main__":
    main()
