import collections
import dataclasses
import datetime
import hashlib
import json
import multiprocessing
import os
import queue
import random
import shutil
import threading
import traceback
import uuid
from multiprocessing import managers
from multiprocessing import shared_memory
from queue import Empty
from typing import List, Callable, Optional, Tuple, Dict
import subprocess
import ffmpeg
import jax
import numpy as np
import requests
import transformers
import urllib3.exceptions
import yt_dlp as youtube_dl

_DEBUG = False
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
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore,
                   target_image_size: int) -> List[dict]:
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

        if ('automatic_captions' not in info or "en" not in info["automatic_captions"] or not
        info['automatic_captions']['en'] or "url" not in info['automatic_captions']['en'][0] or not
        info['automatic_captions']['en'][0]["url"]):
            continue

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue
        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url'],
                           "sub_url": info['automatic_captions']['en'][0]['url'], "title": info["title"]})

    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))


def get_proxies():
    while True:
        try:
            r = requests.get("https://proxy.webshare.io/api/proxy/list/?mode=backbone&page_size=1000",
                             headers={"Authorization": os.environ["WEBSHARE_KEY"]})
            return [f"{r['username']}:{r['password']}" + '@' + f"{r['proxy_address']}:{r['ports']['socks5']}" for r in
                    r.json()['results']]
        except:
            pass


def get_sentences(subtitles, fps):
    sentences = []
    timestamps = []
    timestamp = 0

    for subtitle in subtitles:
        if 'aAppend' in subtitle:
            timestamp = subtitle['tStartMs']
            sentences.append(' ')
            timestamps.append(timestamp)

        if 'segs' in subtitle:
            segs = subtitle['segs']
            start = subtitle['tStartMs']
            for seg in segs:
                word = seg['utf8']
                if 'tOffsetMs' in seg:
                    start += seg['tOffsetMs']
                sentences.append(word)
                timestamps.append(timestamp)

    if not sentences:
        return np.array([]), np.array([])

    out_s = [sentences[0]]
    out_t = [timestamps[0]]
    for s, t0, t1 in zip(sentences[1:], timestamps[:-1], timestamps[1:]):
        if t0 == t1:
            out_s[-1] = f"{out_s[-1]} {s}"
        else:
            out_s.append(s)
            out_t.append(round(fps * t1 / 1000))
    out_s = np.array(out_s)
    out_t = np.array(out_t)
    return out_s, out_t

@try_except
def get_subs(video_urls: List[Dict[str, str]], proxies: List[str], target_fps: int):
    while True:
        for _ in range(len(proxies)):
            p = proxies.pop(0)
            try:
                with open("subs_urls.txt", "w") as f:
                    f.write(str(video_urls))
                subs = requests.get(video_urls[0]["sub_url"],
                                    proxies={"http": f"socks5://{p}", "https": f"socks5://{p}"}).text
                events = json.loads(subs)['events']
                return get_sentences(events, target_fps)

            except urllib3.exceptions.HTTPError:
                pass
            except requests.exceptions.RequestException:
                pass
            except json.decoder.JSONDecodeError:
                pass

            print('error')

        proxies.clear()
        proxies.extend(get_proxies())
        print("Refreshing proxies", len(proxies))


def get_video_frames(video_urls: List[dict], target_image_size: int, target_fps: int, device_steps: int, context: int) -> np.ndarray:
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
        w = round(target_image_size * aspect_ratio) if aspect_ratio > 1 else target_image_size
        h = target_image_size if aspect_ratio > 1 else round(target_image_size / aspect_ratio)
        try:
            vid = ffmpeg.input(path)
            vid = vid.filter("scale", w=w, h=h)
            vid = vid.filter("crop", w=target_image_size, h=target_image_size)
            vid = vid.filter("fps", target_fps)
            vid = vid.filter("select", f"between(n,{0},{8*device_steps*context})")
            vid = vid.output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error", preset="ultrafast",
                             threads=target_image_size // 40)
            out, _ = vid.run(capture_stdout=True)
        except ffmpeg.Error:  # Broken Video, next might work
            continue

        if os.path.exists(path):
            os.remove(path)
        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 context_size: int, queue: multiprocessing.Queue, smm: managers.SharedMemoryManager,
                 ip_addresses: List[str], device_steps: int):
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
        {'writeautomaticsub': True, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
         "ignoreerrors": True})
    youtube_getter.add_default_info_extractors()
    rng = random.Random(worker_id)
    rng.shuffle(work)

    group = context_size * device_steps
    while True:
        for i, wor in enumerate(work, worker_id):
            video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)

            if not video_urls:
                continue

            subtitles = get_subs(video_urls, ip_addresses, target_fps)

            if subtitles is None:
                continue

            frames = get_video_frames(video_urls, target_image_size, target_fps, device_steps, context_size)

            if frames is None or not frames.size or frames.shape[0] < group:
                continue
                
            subs, timestamps = subtitles
            timed_subs = []
            for frame_idx in range(len(frames)):
                timed_subs.append(subs[timestamps <= frame_idx][-1])
            timed_subs = np.array(timed_subs)

            timed_subs = timed_subs[:timed_subs.shape[0] // group * group]
            timed_subs = timed_subs.reshape(-1, context_size, *timed_subs.shape[1:])

            batch_timed_subs = ["".join(list(dict.fromkeys(sub_list))) for sub_list in timed_subs]

            frames = frames[:frames.shape[0] // group * group]
            frames = frames.reshape(-1, context_size, *frames.shape[1:])[:8 * device_steps]

            queue.put((to_share(frames, smm), batch_timed_subs))
        queue.put(_DONE)


class DataLoader:
    def __init__(self, workers: int, url_dir: str, video_downloaders: int, resolution: int, fps: int, context: int,
                 batch_size: int, prefetch: int, parallel_videos: int, tokenizer: transformers.BertTokenizer,
                 clip_tokens: int, device_steps: int, batch_prefetch: int, seed: int = 0):
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
        self.clip_tokens = clip_tokens
        self.ids = ids = []
        self.device_steps = device_steps
        for path in os.listdir(url_dir):
            with open(f'{url_dir}/{path}', 'rb') as f:
                vals = json.load(f)
                ids.extend([x for i, d in zip(vals["id"], vals["duration"]) for x, z in zip(i, d) if
                            z > context * self.device_steps / fps])
        self.rng = random.Random(self.seed)
        self.rng.shuffle(self.ids)
        self.ids = ids[int(len(ids) * jax.process_index() / jax.process_count()):
                       int(len(ids) * (jax.process_index() + 1) / jax.process_count())]

        self.running = False
        self.batch_queue = queue.Queue(batch_prefetch)
        self.thread: Optional[threading.Thread] = None
        self.batch_thread: Optional[threading.Thread] = None
        self._start()

    def _start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
        return

    def _batch(self, samples: List[Tuple[List[np.ndarray], str]]):
        idx = 0
        np_batch = []
        subs = []
        while self.running:
            if len(np_batch) == self.batch_size:
                if _DEBUG:
                    self.batch_queue.put([hashlib.sha3_512(s.encode()).hexdigest() for s in subs])
                    continue

                input_ids = []
                attention_mask = []

                for sub in subs:
                    tokens = self.tokenizer(sub.tolist(), return_tensors="np", padding="max_length", truncation=True,
                                            max_length=self.clip_tokens)
                    input_ids.append(np.stack(tokens["input_ids"]))
                    attention_mask.append(np.stack(tokens["attention_mask"]))

                self.batch_queue.put((np.stack(np_batch), np.stack(input_ids), np.stack(attention_mask)))
                np_batch.clear()
                subs.clear()

            while len(samples) > idx and not samples[idx][0]:
                del samples[idx]
            if len(samples) <= idx:
                continue
            np_batch.append(np.stack([samples[idx][0].pop(0) for _ in range(self.device_steps)]))
            subs.append(np.stack([samples[idx][1].pop(0) for _ in range(self.device_steps)]))

            idx = (idx + 1) % self.parallel_videos

    def _worker(self):
        self.rng.shuffle(self.ids)
        lock = multiprocessing.Semaphore(self.video_downloaders)
        queue = multiprocessing.Queue(self.prefetch)
        samples = []
        workers = []
        self.batch_thread = threading.Thread(target=self._batch, args=(samples,))
        self.batch_thread.start()
        done = 0
        with managers.SharedMemoryManager() as smm:
            proxies = get_proxies()
            for i in range(self.workers):
                work = self.ids[int(len(self.ids) * i / self.workers):int(len(self.ids) * (i + 1) / self.workers)]
                args = work, i, lock, self.resolution, self.fps, self.context, queue, smm, proxies, self.device_steps
                workers.append(multiprocessing.Process(args=args, daemon=True, target=frame_worker))
            for w in workers:
                w.start()
            while True:
                if done == self.workers:
                    break
                if len(samples) >= self.parallel_videos:
                    continue
                try:
                    out = queue.get(timeout=120)
                except Empty:
                    print(datetime.datetime.now(), f"Queue empty. Waiting another 120 seconds. Already have {len(samples)} videos.")
                    continue
                if out == _DONE:
                    done += 1
                    continue

                try:
                    share = list(from_share(out[0]))  # this also removes share from_shared_memory
                    # self.rng.shuffle(share)
                    samples.append((share, out[1]))
                except:
                    print("failed to load share")
                continue

            for w in workers:
                w.join()
        self.running = False

    def __iter__(self):
        self._start()
        while self.running:
            try:
                yield self.batch_queue.get(timeout=60)
            except queue.Empty:
                continue
        raise StopIteration


if __name__ == '__main__':
    sub_hashes = collections.defaultdict(int)
    for i in DataLoader(1, "/home/ubuntu/urls/", 1, 8, 1, 120, 1, 1, 128,
                        transformers.AutoTokenizer.from_pretrained("google/long-t5-local-base"), 1):
        for h in i:
            sub_hashes[h] += 1
        print({h[:6]: sub_hashes[h] for h in i})

