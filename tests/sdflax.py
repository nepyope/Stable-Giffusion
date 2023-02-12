import logging
import threading
import os
import random

import gdown
from PIL import Image
import numpy as np
import torch
import torch.utils.checkpoint
import json
import jax
import jax.numpy as jnp
import optax
import transformers
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from flax.jax_utils import replicate
import wandb
import cv2
import requests
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard

from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel, set_seed   


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = logging.getLogger(__name__)

def main():

    # create the directory if it does not exist
    if not os.path.exists("surl"):
        os.makedirs("surl")
        url = 'https://drive.google.com/uc?id=1fK3B7B9gzQDp14xfNvkyaIuUr36QEJrV'
        output = 'surl/10M_train.json'
        gdown.download(url, output, quiet=False)


    resolution = (256,512)
    batch_per_device = 2
    lr = 1e-5

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(0)

    # Handle the repository creation
    if jax.process_index() == 0:
        os.makedirs('sd-model', exist_ok=True)


    train_transforms = transforms.Compose(
        [   transforms.CenterCrop(resolution),
            #transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    weight_dtype = jnp.float32

    # Load models and create wrapper for stable diffusion
    model_path = 'flax/stable-diffusion-2-1'
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_path, subfolder="vae", dtype=weight_dtype
    )


    constant_scheduler = optax.constant_schedule(lr)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=0.9,
        b2=0.999,
        eps=1e-08,
        weight_decay=1e-2,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1),
        adamw,
    )

    vae_state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)

    def vae_train_step(state, batch):

        def compute_loss(params):
            inp = batch["pixel_values"][0]
            inp = jnp.expand_dims(inp, axis=0)
            tar = batch["pixel_values"][1]
            tar = jnp.expand_dims(tar, axis=0)
            gaussian, dropout = jax.random.split(jax.random.PRNGKey(0), 2)
            out = vae.apply({"params": params}, inp, rngs={"gaussian": gaussian, "dropout": dropout},
                            sample_posterior=True, deterministic=False).sample

            loss = (tar - out) ** 2

            loss = loss.mean()

            return loss

        #do separate trainning for vae
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    vae_p_train_step = jax.pmap(vae_train_step, "batch", donate_argnums=(0,))

    vae_state = jax_utils.replicate(vae_state)

    vae_params = jax_utils.replicate(vae_params)

    ####LOAD DATA
    url_dir = 'surl'
    for path in os.listdir(url_dir):
        with open(f'{url_dir}/{path}', 'rb') as f:
            vals = json.load(f)
            ids = [x for x in list(zip(vals["url"], vals["duration"], vals["text"])) if x[1] >= 11 and x[1]<20 and x[2] != ""]
    random.shuffle(ids)
    ids = iter(ids)

    ######TRAIN LOOP
    epochs = 10000000 
    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")
    epochs = tqdm(range(epochs), desc="Epoch ... ", position=0)    

    n_batches = 0
    caption = ""

    def get_data(ids,batch_per_device,data, n_batches, batch_size, caption):
        id = next(ids)
        print('downloading video...')
        url = id[0]

        caption.append([])
        caption[-1].append(id[2])
        caption.pop(0)

        r = requests.get(url, allow_redirects=True)
        open('video.mp4', 'wb').write(r.content)

        video = cv2.VideoCapture('video.mp4')

        # Get the total number of frames in the video
        total_frames = 256

        data.append([])

        for i in range(0, total_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                data[-1].append(frame)

        data.pop(0)

        batch_size.append([])
        batch_size[-1].append(batch_per_device*jax.device_count())
        batch_size.pop(0)

        n_batches.append([])
        n_batches[-1].append(len(data[0]) // batch_size[0][0])
        n_batches.pop(0)

    new_data = [[]]
    new_n_batches = [[]]
    new_batch_size = [[]]
    new_caption = [[]]

    fetch = threading.Thread(target=get_data, name="Downloader", args=(ids,batch_per_device,new_data, new_n_batches, new_batch_size, new_caption))
    fetch.start()
    fetch.join()

    data, n_batches, batch_size, caption = new_data[0], new_n_batches[0][0], new_batch_size[0][0], new_caption[0][0] #overfit vae on this one example

    for epoch in epochs:

        #fetch = threading.Thread(target=get_data, name="Downloader", args=(ids,batch_per_device,new_data, new_n_batches, new_batch_size, new_caption))
        #fetch.start()
        #print(caption)
        for shift in range(10):#shift batch y 1 so that all transitions are learned 

            iters = tqdm(range(n_batches), desc="Iter ... ", position=1)

            for i in iters:#maybe repeat this multiple times, sample afterwards

                ####LOAD DATA
                d = data[i*batch_size:(i+1)*batch_size]

                d = d[shift%2:] + d[:shift%2]#this is done so that the transition is learned from frame 0 to 1, 1 to 2, 2 to 3.. instead of 0 to 1, 2 to 3, 4 to 5
                

                images = []
                captions = []
                for n,image in enumerate(d):
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(img))
                    captions.append(f'{caption}')

                #append captions[0] to text file
                with open('text.txt', 'a') as file:
                    file.write(f'{caption[0]}')

                inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
                input_ids = inputs.input_ids
                images = [image.convert("RGB") for image in images]
                images = [train_transforms(image) for image in images]

                examples = []
                for i in range(len(images)):
                    example = {}
                    example["pixel_values"] = (torch.tensor(images[i])).float()
                    example["input_ids"] = input_ids[i]
                    examples.append(example)
                pixel_values = torch.stack([example["pixel_values"] for example in examples])


                pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
                input_ids = [example["input_ids"] for example in examples]

                padded_tokens = tokenizer.pad(
                    {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
                )
                batch = {
                    "pixel_values": pixel_values,
                    "input_ids": padded_tokens.input_ids,
                }
                batch = {k: v.numpy() for k, v in batch.items()}

                batch = {"pixel_values": batch['pixel_values'].reshape(jax.local_device_count(), -1, *batch['pixel_values'].shape[1:]),
                        "input_ids": batch['input_ids'].reshape(jax.local_device_count(), -1, *batch['input_ids'].shape[1:])}

                vae_state, vae_loss = vae_p_train_step(vae_state, batch)
                vae_loss = sum(vae_loss['loss'])/jax.device_count()

                run.log({"VAE loss": vae_loss})

        if epoch % 100 == 0:#save every 10 epochs
            #generate samples using stuff i've already loaded 
            if jax.process_index() == 0:#need to work on this, it has to cylcle a bunch in order to work 
                print('saving model...')
                vae.save_pretrained('vae', params=vae_state.params)
                print('resuming training')      

        #fetch.join()
        #data, n_batches, batch_size, caption = new_data[0], new_n_batches[0][0], new_batch_size[0][0], new_caption[0][0]

if __name__ == "__main__":
    main()
