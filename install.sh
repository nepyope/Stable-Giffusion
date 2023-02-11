python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]==0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
python3 -m pip install wandb smart-open[gcs] jsonpickle sharedutils git+https://github.com/ytdl-org/youtube-dl/ typer diffusers flax optax ffmpeg-python huggingface-hub transformers gdown torch torchvision opencv-python ftfy
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 protobuf==3.20.1
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]==0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
