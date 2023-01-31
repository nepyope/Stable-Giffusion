import collections
import dataclasses
import datetime
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import threading
import traceback
import uuid
from multiprocessing import managers
from multiprocessing import shared_memory
from queue import Empty
from typing import List, Callable, Tuple, Dict
import whisper
import torch
import time
import ffmpeg
import ftfy
#import jax
import numpy as np
import requests
import transformers
import urllib3.exceptions
import youtube_dl
import psutil
_DEBUG = False
_DONE = "DONE"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclasses.dataclass
class Share:
    dtype: np.dtype
    shape: List[int]
    name: str


def to_share(inp: np.array, smm: managers.SharedMemoryManager) -> Share:
    mem = smm.SharedMemory(inp.nbytes)
    np_mem = np.ndarray(inp.shape, dtype=inp.dtype, buffer=mem.buf)
    np_mem[:] = inp[:]
    return Share(dtype=inp.dtype, shape=inp.shape, name=mem.name)


def from_share(share: Share) -> np.ndarray:
    mem = shared_memory.SharedMemory(name=share.name, create=False)
    arr = np.copy(np.ndarray(share.shape, share.dtype, buffer=mem.buf))
    mem.unlink()
    return arr


def try_except(fn: Callable, default=None):
    def _fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # skipcq: PYL-W0703
            print(r"IGNORED EXCEPTION \/\/\/")
            print(fn, exc)
            traceback.print_exc()
            print("IGNORED EXCEPTION /\\/\\/\\")

        return default

    return _fn


@try_except
def get_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore, target_image_size: int
                   ) -> List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
    # the same time.

    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    audio_urls = []
    for f in info['formats']:
        if f.get('acodec') != 'none' and f.get('vcodec') == 'none':
            audio_urls.append({'ext': f['ext'], 'url': f['url'], 'tbr': f.get('tbr')})

        width = f.get('width')
        height = f.get('height')
        url = f.get('url')
        ext = f.get('ext')
        format_note = f.get('format_note')

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue

        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url'],
                           })

    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height'])), sorted(audio_urls, key=lambda x: x['tbr'])


def get_video(video_urls: List[dict], target_image_size: int, target_fps: int,
                     ) -> np.ndarray:
    filename = uuid.uuid4()
    path = str(filename)
    for video_url in video_urls:
        if os.path.exists(path):
            os.remove(path)

        url = video_url["url"]
        path = f"{filename}.{video_url['ext']}"
        try:
            with requests.get(url, stream=True) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        except Exception:  # skipcq: PYL-W0703
            continue  # Broken URL, next might work
        width = round(video_url["width"] * video_url["height"] / target_image_size)
        try:
            out, _ = ffmpeg.input(path) \
                .filter("scale", w=width, h=target_image_size) \
                .filter("crop", w=target_image_size, h=target_image_size).filter("fps", target_fps) \
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error", preset="ultrafast",
                        threads=target_image_size // 40) \
                .run(capture_stdout=True)
        except ffmpeg.Error:  # Broken Video, next might works
            continue
        
        if os.path.exists(path):
            os.remove(path)

        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))

def get_subs(audio_urls: List[dict]) -> str:
    filename = uuid.uuid4()
    path = str(filename)

    model_fp32 = whisper.load_model(
    name="base",
    device="cpu")
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    print('loaded model')
    for audio_url in audio_urls:
        print('downloading audio')
        if os.path.exists(path):
            os.remove(path)

        url = audio_url["url"]
        path = f"{filename}.{audio_url['ext']}"
        try:
            with requests.get(url, stream=True) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        except Exception:  # skipcq: PYL-W0703
            continue  # Broken URL, next might work
        try:
            audio = whisper.load_audio(path)
            print('loaded audio')
            t = time.time()
            subs = whisper.transcribe(quantized_model, audio)['text']
            print(time.time() - t)
            print(subs)
        except:
            continue

        if os.path.exists(path):
            os.remove(path)

        return subs


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 context_size: int, queue: multiprocessing.Queue, smm: managers.SharedMemoryManager,
                 ):

    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
        {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
         "ignoreerrors": True
         })
    youtube_getter.add_default_info_extractors()
    rng = random.Random(worker_id)
    rng.shuffle(work)

    for i, wor in enumerate(work, worker_id):
        video_urls, audio_urls = get_urls(youtube_getter, youtube_base, wor, lock, target_image_size)
        if not video_urls or not audio_urls:
            continue

        frames = get_video(video_urls, target_image_size, target_fps)
        if frames is None or not frames.size or frames.shape[0] < context_size:
            continue
        
        subs = get_subs(audio_urls)
        frames = frames[:frames.shape[0] // context_size * context_size]
        frames = frames.reshape(-1, context_size, *frames.shape[1:])
        queue.put((to_share(frames, smm), subs))
    queue.put(_DONE)


class DataLoader:
    def __init__(self, workers: int, url_dir: str, video_downloaders: int, resolution: int, fps: int, context: int,
                 batch_size: int, prefetch: int, parallel_videos: int, tokenizer: transformers.BertTokenizer,
                 t5_tokens: int=4096, seed: int = 0):
        self.workers = workers
        self.video_downloaders = video_downloaders
        self.resolution = resolution
        self.fps = fps
        self.context = context
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.seed = seed
        self.parallel_videos = parallel_videos
        self.tokenizer = tokenizer

        self.ids = ids = []
        self.t5_tokens = t5_tokens
        for path in os.listdir(url_dir):
            with open(f'{url_dir}/{path}', 'rb') as f:
                vals = json.load(f)
                ids.extend([x for i, d in zip(vals["id"], vals["duration"]) for x, z in zip(i, d) if z > context / fps])
        random.Random(self.seed).shuffle(self.ids)
        #self.ids = ids[int(len(ids) * jax.process_index() / jax.process_count()):
                       #int(len(ids) * (jax.process_index() + 1) / jax.process_count())]

    def __iter__(self):
        random.Random(self.seed).shuffle(self.ids)
        lock = multiprocessing.Semaphore(self.video_downloaders)
        queue = multiprocessing.Queue(self.prefetch)
        workers = []

        with managers.SharedMemoryManager() as smm:

            for i in range(self.workers):
                work = self.ids[int(len(self.ids) * i / self.workers):int(len(self.ids) * (i + 1) / self.workers)]
                args = work, i, lock, self.resolution, self.fps, self.context, queue, smm
                workers.append(multiprocessing.Process(args=args, daemon=True, target=frame_worker))

            for w in workers:
                w.start()
            done = 0
            samples = []
            idx = 0
            while True:

                if done == self.workers:
                    break
                if len(samples) <= idx + self.batch_size and len(samples) < self.parallel_videos:
                    try:
                        out = queue.get(timeout=120)
                    except Empty:
                        print(datetime.datetime.now(), "Queue empty. Waiting another 120 seconds.")
                        continue
                    if out == _DONE:
                        done += 1
                        continue
                    samples.append((list(from_share(out[0])), out[1]))

                else:
                    np_batch = []
                    subtitles = []
                    for _ in range(self.batch_size):
                        while len(samples) > idx and not samples[idx][0]:
                            del samples[idx]
                        if len(samples) <= idx:
                            break
                        np_batch.append(samples[idx][0].pop(0))
                        subtitles.append(samples[idx][1])
                        idx = (idx + 1) % self.parallel_videos
                    if len(np_batch) == self.batch_size:
                        if _DEBUG:
                            yield [hashlib.sha3_512(s.encode()).hexdigest() for s in subtitles]
                            continue
                        tokens = self.tokenizer(subtitles, return_tensors="np", padding="max_length", truncation=True,
                                                max_length=self.t5_tokens)
                        input_ids = tokens["input_ids"].reshape(8 * self.batch_size, -1)
                        attention_mask = tokens["attention_mask"].reshape(8 * self.batch_size, -1)
                        yield np.concatenate(np_batch, axis=0), input_ids, attention_mask
            for w in workers:
                w.join()
        raise StopIteration


if __name__ == '__main__':
    sub_hashes = collections.defaultdict(int)
    for i in DataLoader(1, "/home/ubuntu/urls/", 1, 8, 1, 120, 1, 1, 128,
                        transformers.AutoTokenizer.from_pretrained("google/long-t5-local-base"), 1):
        for h in i:
            sub_hashes[h] += 1
        print({h[:6]: sub_hashes[h] for h in i})
