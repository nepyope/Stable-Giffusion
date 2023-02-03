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

import ffmpeg
import ftfy
import jax
import numpy as np
import requests
import transformers
import urllib3.exceptions
import youtube_dl

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
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore, target_image_size: int
                   ) -> List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
    # the same time.

    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    for f in info['formats']:
        width = f.get('width')
        height = f.get('height')
        url = f.get('url')
        ext = f.get('ext')
        format_note = f.get('format_note')

        if ('automatic_captions' not in info or "en" not in info["automatic_captions"]
                or not info['automatic_captions']['en'] or "url" not in info['automatic_captions']['en'][0]
                or not info['automatic_captions']['en'][0]["url"]):
            continue

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue
        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url'],
                           "sub_url": info['automatic_captions']['en'][0]['url']})

    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))


def get_proxies():
    while True:
        try:
            r = requests.get("https://proxy.webshare.io/api/proxy/list/?mode=backbone&page_size=1000",
                             headers={"Authorization": "sudl99m2vcl0kf6x1wyh7pa8nnatg378lo9ltwct"})
            return [f"{r['username']}:{r['password']}" + '@' + f"{r['proxy_address']}:{r['ports']['socks5']}"
                    for r in r.json()['results']]
        except:
            pass


@try_except
def get_subs(video_urls: List[Dict[str, str]], proxies: List[str]):
    while True:
        for _ in range(len(proxies)):
            p = proxies.pop(0)
            try:
                subs = requests.get(video_urls[0]["sub_url"],
                                    proxies={"http": f"socks5://{p}", "https": f"socks5://{p}"}).text
                subs = subs[subs.find("<transcript>") + len("<transcript>"):subs.find('</text>')]
                subs = subs[subs.find('>') + 1:]
                subs = ftfy.ftfy(subs)
                if "but your computer or network may be sending automated queries. To protect our users, we can't process your request right now." in subs:
                    print("blocked IP")
                    continue
                proxies.append(p)
                return subs
            except urllib3.exceptions.HTTPError:
                pass
            except requests.exceptions.RequestException:
                pass
            print("error")
        proxies.clear()
        proxies.extend(get_proxies())
        print("Refreshing proxies", len(proxies))


def get_video_frames(video_urls: List[dict], target_image_size: int, target_fps: int,
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
        aspect_ratio = video_url["width"] / video_url["height"]
        try:
            out, _ = ffmpeg.input(path) \
                .filter("scale", w=round(target_image_size*aspect_ratio), h=target_image_size) \
                .filter("crop", w=target_image_size, h=target_image_size).filter("fps", target_fps) \
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error", preset="ultrafast",
                        threads=target_image_size // 40) \
                .run(capture_stdout=True)
        except ffmpeg.Error:  # Broken Video, next might work
            continue

        if os.path.exists(path):
            os.remove(path)
        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 context_size: int, queue: multiprocessing.Queue, smm: managers.SharedMemoryManager,
                 ip_addresses: List[str]):
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
        {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
         "ignoreerrors": True
         })
    youtube_getter.add_default_info_extractors()
    rng = random.Random(worker_id)
    rng.shuffle(work)

    for i, wor in enumerate(work, worker_id):
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)
        if not video_urls:
            continue

        subs = get_subs(video_urls, ip_addresses)
        if not subs:
            continue

        frames = get_video_frames(video_urls, target_image_size, target_fps)
        if frames is None or not frames.size or frames.shape[0] < context_size:
            continue
        
        frames = frames[:frames.shape[0] // context_size * context_size]
        frames = frames.reshape(-1, context_size, *frames.shape[1:])
        queue.put((to_share(frames, smm), subs))
    queue.put(_DONE)


class DataLoader:
    def __init__(self, workers: int, url_dir: str, video_downloaders: int, resolution: int, fps: int, context: int,
                 batch_size: int, prefetch: int, parallel_videos: int, tokenizer: transformers.BertTokenizer,
                 t5_tokens: int, seed: int = 0):
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
        self.ids = ids[int(len(ids) * jax.process_index() / jax.process_count()):
                       int(len(ids) * (jax.process_index() + 1) / jax.process_count())]

    def __iter__(self):
        random.Random(self.seed).shuffle(self.ids)
        lock = multiprocessing.Semaphore(self.video_downloaders)
        queue = multiprocessing.Queue(self.prefetch)
        workers = []
        with managers.SharedMemoryManager() as smm:
            proxies = get_proxies()
            for i in range(self.workers):
                work = self.ids[int(len(self.ids) * i / self.workers):int(len(self.ids) * (i + 1) / self.workers)]
                args = work, i, lock, self.resolution, self.fps, self.context, queue, smm, proxies
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
