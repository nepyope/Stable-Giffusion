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


    resolution = (256,256)
    batch_per_device = 1
    lr = 1e-4/(batch_per_device*jax.device_count())#i guess

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
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", dtype=weight_dtype
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_path, subfolder="vae", dtype=weight_dtype
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", dtype=weight_dtype
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

    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    vae_state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler.create_state()

    # Initialize our training
    rng = jax.random.PRNGKey(0) 
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def unet_train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"],
                params=text_encoder_params
            )[0]

            # Predict the noise residual and compute loss
            model_pred = unet.apply(
                {"params": params}, noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (target - model_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng

    def vae_train_step(state, batch):

        def compute_loss(params):
            gaussian, dropout = jax.random.split(jax.random.PRNGKey(0), 2)
            img = batch["pixel_values"]
            out = vae.apply({"params": params}, img, rngs={"gaussian": gaussian, "dropout": dropout},
                            sample_posterior=True, deterministic=False).sample

            loss = (img - out) ** 2
            
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

    # Create parallel version of the train step
    unet_p_train_step = jax.pmap(unet_train_step, "batch", donate_argnums=(0,))
    vae_p_train_step = jax.pmap(vae_train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    unet_state = jax_utils.replicate(unet_state)
    vae_state = jax_utils.replicate(vae_state)
        
    text_encoder_params = jax_utils.replicate(text_encoder.params)
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

    epochs = 10000000 #5 at most? depends on resolution. 5*256*256 seems the max, 2 for widescreen. this is also the number of videos. 
    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")
    epochs = tqdm(range(epochs), desc="Epoch ... ", position=0)    

    n_batches = 0
    caption = ""


    def get_data(ids,batch_per_device,data, n_batches, batch_size, caption):
        id = next(ids)
        print('downloading video...')
        url = id[0]
        duration = id[1]

        caption.append([])
        caption[-1].append(id[2])
        caption.pop(0)


        r = requests.get(url, allow_redirects=True)
        open('video.mp4', 'wb').write(r.content)

        video = cv2.VideoCapture('video.mp4')

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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

    data, n_batches, batch_size, caption = new_data[0], new_n_batches[0][0], new_batch_size[0][0], new_caption[0][0]

    for epoch in epochs:
        
        fetch = threading.Thread(target=get_data, name="Downloader", args=(ids,batch_per_device,new_data, new_n_batches, new_batch_size, new_caption))
        fetch.start()

        for _ in range(10):#repeat training 10 times. how low can i get this?

            iters = tqdm(range(n_batches), desc="Iter ... ", position=1)
            ######UNET TRAINING
            for i in iters:#maybe repeat this multiple times, sample afterwards

                ####LOAD DATA
                d = data[i*batch_size:(i+1)*batch_size]
                images = []
                captions = []
                for n, i in enumerate(d):
                    #load image from cv2
                    img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(img))
                    captions.append(f'{caption} FRAME{n}')

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

                ####TRAIN
                unet_state, unet_loss, train_rngs = unet_p_train_step(unet_state, text_encoder_params, vae_params, batch, train_rngs)
                unet_loss = sum(unet_loss['loss'])/jax.device_count()

                run.log({"UNET loss": unet_loss})

            iters = tqdm(range(n_batches), desc="Iter ... ", position=1)
            ######VAE TRAINING
            for i in iters:#maybe repeat this multiple times, sample afterwards

                ####LOAD DATA
                d = data[i*batch_size:(i+1)*batch_size]
                images = []
                captions = []
                for n, i in enumerate(d):
                    #load image from cv2
                    img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(img))
                    captions.append(f'{caption} FRAME{n}')

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

            if jax.process_index() == 0:#need to work on this, it has to cylcle a bunch in order to work 
                print('saving model...')
                scheduler = FlaxPNDMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                )
                safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker", from_pt=True
                )
                pipeline = FlaxStableDiffusionPipeline(
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                    safety_checker=safety_checker,
                    feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                )

                pipeline.save_pretrained(
                    'sd-model',
                    params={
                        "text_encoder": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], text_encoder_params)),
                        "vae": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], vae_state.params)),
                        "unet": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], unet_state.params)),
                        "safety_checker": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], safety_checker.params)),
                    },
                )
                print('generating samples...')
                pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
                    "sd-model",
                    dtype=weight_dtype,
                    safety_checker=None,
                )
                prompt = []
                prompt = [f'{caption} FRAME0']*jax.device_count()
                prompt_ids = pipeline.prepare_inputs(prompt)
                p_params = replicate(params)
                prompt_ids = shard(prompt_ids)
                def create_key(seed=0):
                    return jax.random.PRNGKey(seed)

                rng = create_key(0)
                rng = jax.random.split(rng, jax.device_count())

                images = pipeline(prompt_ids, p_params, rng, jit=True,width=resolution[1],height=resolution[0])[0]    

                #send images to wandb
                run.log({"prompt": wandb.Html(f'<h1>{prompt[0]}</h1>')})
                run.log({f"result": wandb.Image(images[0])})
                run.log({"ground_truth": wandb.Image(batch["pixel_values"][0][0].transpose([1,2,0]))})

                #get rid of pipeline
                del pipeline
                del params
                del p_params
                del prompt_ids
                del images
                del prompt
                del rng
                del safety_checker
                del scheduler
                print('resuming training')      
        
        fetch.join()
        data, n_batches, batch_size, caption = new_data[0], new_n_batches[0][0], new_batch_size[0][0], new_caption[0][0]

if __name__ == "__main__":
    main()
