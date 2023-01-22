import dataclasses
import datetime
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
from typing import List, Callable

import ffmpeg
import jax
import numpy as np
import requests
import youtube_dl
import json
import random
_DONE = "DONE"


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
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore, target_image_size: int, ip_addresses: list) -> \
        List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
    # the same time.


    proxy = random.randint(0, len(ip_addresses) - 1)
    proxies = {"http": f"socks5://{ip_addresses[proxy]}", "https": f"socks5://{ip_addresses[proxy]}"}
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


        if not 'automatic_captions' in info:
            continue

        url = info['automatic_captions']['en'][4]['url']
        print(info['automatic_captions']['en'])
        vtt = requests.get(url, proxies=proxies).text
        #subs = decode_vtt(vtt)

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue
        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url']})
    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))



def decode_vtt(content: str):
    '''
    :param content: String with the of the .vtt file.
    :return: String with combined text, List with Strings split at time stamps and List with float second time stamps.
    This Function decodes a vtt to get the contend with  time stamps.
    '''

    if '</c><' in content and '><c>' in content:

        # Split the content at line brake and check if word level time stamps are in the line.
        content = [l for l in content.split('\n') if '<c>' in l]

        # Connect list of strings back together.
        content = "".join(content)

        # Split String at time stamp headers.
        content = content.split('><c>')

        # Create output lists.
        words = []
        stamps = []

        # Loop word and time stamp string list.
        for c in content:

            # Split word and time stamp part.
            word = c[:-12]
            stam = c[-12:]

            # Clean word string.
            if not '</c><' in word:
                word = word.replace('</c>', ' ')

            word = word.replace('</c>', '')
            word = word.replace('<', '')
            word = word.lstrip().rstrip()

            # Check if time stamp is in stamp string.
            if ':' in stam and '.' in stam:

                # Split time stamp string at time punctuation marks.
                stam = stam.split(':')
                stam = stam[:-1] + stam[-1].split('.')

                # Converting time stamp string in to second based float.
                stam = datetime.timedelta(hours=int(stam[0]), minutes=int(stam[1]), seconds=int(stam[2]), milliseconds=int(stam[3]))
                stam = stam.total_seconds()

                # add word string and second based float to output list.
                words.append(' ' + word)
                stamps.append(stam)
            else:
                # If no time stamp contain in time stamp part we assume that it is a another word.
                # If it as a another word we add it to the previous word string.
                if len(words) > 0:
                    words[-1] = words[-1] + " " + c.replace('</c>', '').replace('<', '').lstrip().rstrip()

        return ''.join(words), words, stamps

    else:

        # Split the content at line brake.
        content = content.split('\n')

        # Create output lists.
        words_buffer = []
        stamps_buffer = []
        words = []
        stamps = []

        # Loop word and time stamp string list.
        for idx in range(len(content)):
            if ' --> ' in content[idx]:
                stamps_buffer.append(content[idx])

                word_buffer = []
                idx += 1
                while idx + 1 < len(content) and ' --> ' not in content[idx + 1]:
                    word_buffer.append(content[idx])
                    idx += 1

                words_buffer.append(" ".join(word_buffer))

        for idx in range(len(stamps_buffer)):
            s = stamps_buffer[idx].split(' --> ')

            s_1 = s[0]
            s_1 = s_1.split(':')
            s_1 = s_1[:-1] + s_1[-1].split('.')

            s_2 = s[1]
            s_2 = s_2.split(':')
            s_2 = s_2[:-1] + s_2[-1].split('.')

            s_1 = datetime.timedelta(hours=int(s_1[0]), minutes=int(s_1[1]), seconds=int(s_1[2]),
                                     milliseconds=int(s_1[3]))
            s_1 = s_1.total_seconds()

            s_2 = datetime.timedelta(hours=int(s_2[0]), minutes=int(s_2[1]), seconds=int(s_2[2]),
                                     milliseconds=int(s_2[3]))
            s_2 = s_2.total_seconds()

            stamps_buffer[idx] = [s_1, s_2]

        for idx in range(len(words_buffer)):
            word = words_buffer[idx].lstrip().rstrip()
            wor = [' ' + w for w in word.split(' ')]

            stamp = stamps_buffer[idx]

            time_snip = (stamp[1] - stamp[0]) / len(wor)

            stamps += [stamp[0] + i * time_snip for i in range(len(wor))]
            words += wor

        return ''.join(words), words, stamps


def get_video_frames(video_urls: List[dict], target_image_size: int, target_fps: int) -> np.ndarray:
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
        except ffmpeg.Error:  # Broken Video, next might work
            continue

        if os.path.exists(path):
            os.remove(path)
        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 context_size: int, queue: multiprocessing.Queue, smm: managers.SharedMemoryManager):
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
        {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
         "ignoreerrors": True
         })
    youtube_getter.add_default_info_extractors()
    random.Random(worker_id).shuffle(work)

    r = requests.get("https://proxy.webshare.io/api/proxy/list/", headers={"Authorization": "wt7c6034fy30k5gk14jlacqh0xflh8j4x7a5lcut"})
    #append ips to a proxy list
    ip_addresses  = []
    for r in r.json()['results']:
        p = f"{r['username']}:{r['password']}"+'@'+f"{r['proxy_address']}:{r['ports']['socks5']}"
        ip_addresses.append(p)



    for wor in work:
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size, ip_addresses)

        if not video_urls:
            continue

        frames = get_video_frames(video_urls, target_image_size, target_fps)

        if frames is None or not frames.size or frames.shape[0] < context_size:
            continue
        frames = frames[:frames.shape[0] // context_size * context_size]
        frames = frames.reshape(-1, context_size, *frames.shape[1:])
        queue.put(to_share(frames, smm))
    queue.put(_DONE)


class DataLoader:
    def __init__(self, workers: int, url_dir: str, video_downloaders: int, resolution: int, fps: int, context: int,
                 batch_size: int, prefetch: int, parallel_videos: int, seed: int = 0):
        self.workers = workers
        self.video_downloaders = video_downloaders
        self.resolution = resolution
        self.fps = fps
        self.context = context
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.seed = seed
        self.parallel_videos = parallel_videos
        self.ids = ids = []
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
                if len(samples) < self.parallel_videos:
                    try:
                        out = queue.get(timeout=120)
                    except Empty:
                        print(datetime.datetime.now(), "Queue empty. Waiting another 120 seconds.")
                        continue
                    if out == _DONE:
                        done += 1
                        continue
                    samples.append(list(from_share(out)))
                else:
                    batch = []
                    for _ in range(self.batch_size):
                        idx = (idx + 1) % self.parallel_videos
                        while not samples[idx]:
                            del samples[idx]
                        batch.append(samples[idx].pop(0))
                    yield np.concatenate(batch, axis=0)
            for w in workers:
                w.join()
        raise StopIteration
