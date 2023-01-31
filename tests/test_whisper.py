import os


import typer

from diffusers.utils import check_min_version

import tqdm

from transformers import AutoTokenizer
from data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")

_CONTEXT = 0
_KERNEL = 3
_RESHAPE = False

@app.command()
def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-16,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1",
         kernel: int = 3, data_path: str = "./urls", batch_size: int = 4,
         sample_interval: int = 64, parallel_videos: int = 128,
         tracing_start_step: int = 128, tracing_stop_step: int = 196,
         schedule_length: int = 1024,
         guidance: float = 7.5,
         unet_batch_factor: int = 16,
         warmup_steps: int = 2048,
         lr_halving_every_n_steps: int = 8192):
    global _KERNEL, _CONTEXT, _RESHAPE
    _CONTEXT, _KERNEL = context, kernel

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")

    tokenizer: AutoTokenizer = tokenizer


    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, batch_size, prefetch, parallel_videos,
                      tokenizer)
    for i, (video, input_ids, attention_mask) in tqdm.tqdm(enumerate(data)):
        print(video.shape, input_ids.shape, attention_mask.shape)
        if i == 10:
            break


if __name__ == "__main__":
    app()
