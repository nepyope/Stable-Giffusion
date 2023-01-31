python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo python3 -m pip uninstall tensorboard tbp-nightly tb-nightly tensorboard-plugin-profile -y
python3 -m pip install wandb smart-open[gcs] jsonpickle sharedutils git+https://github.com/ytdl-org/youtube-dl/ typer diffusers flax optax ffmpeg-python huggingface-hub transformers gdown torch torchvision opencv-python ftfy
python3 -m pip install --upgrade --force-reinstall tensorflow==2.8.0 protobuf==3.20.1

wget https://files.pythonhosted.org/packages/a3/50/c4d2727b99052780aad92c7297465af5fe6eec2dbae490aa9763273ffdc1/pip-22.3.1.tar.gz
tar -xzvf pip-22.3.1.tar.gz
cd pip-22.3.1
python3 -m setup.py install
cd .. 

git clone https://github.com/MiscellaneousStuff/openai-whisper-cpu
cd openai-whisper-cpu
git submodule init
git submodule update
pip install -e ./whisper
cd .. 
sudo rm -r openai-whisper-cpu
