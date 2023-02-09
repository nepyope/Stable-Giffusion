from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import smart_open
import typer
from PIL import Image
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.utils import check_min_version
import flax
from flax import linen as nn
from jax import lax
from jax.experimental.compilation_cache import compilation_cache
from transformers import AutoTokenizer, FlaxLongT5Model, T5Tokenizer

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
compilation_cache.initialize_cache("compilation_cache")

_CONTEXT = 0
_RESHAPE = False
_UPLOAD_RETRIES = 8

_original_call = nn.Conv.__call__


def conv_call(self: nn.Conv, x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, self.dtype)
    if _RESHAPE and "quant" not in self.scope.name:
        shape = x.shape
        normalizer = jnp.arange(_CONTEXT * 2).reshape(1, -1, *(1,) * (x.ndim - 1)) + 1
        x = x.reshape(-1, _CONTEXT, *shape[1:])
        x0 = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], 1)
        cat = jnp.concatenate([x0, x], 1)
        cat = jnp.concatenate([cat, lax.cumsum(cat, 1) / normalizer], 1)
        x = cat.reshape(-1, 4, *x.shape[1:]).transpose(0, *range(2, x.ndim), 1, x.ndim).reshape(*shape[:-1], -1)
    return _original_call(self, x)


nn.Conv.__call__ = conv_call


def patch_weights(weights: Dict[str, Any], do_patch: bool = False):
    new_weights = {}
    for k, v in weights.items():
        if isinstance(v, dict):
            new_weights[k] = patch_weights(v, ("conv" in k and "quant" not in k) or do_patch)
        elif isinstance(v, (list, tuple)):
            new_weights[k] = list(zip(*sorted(patch_weights(dict(enumerate(v)), "conv" in k or do_patch).items())))[1]
        elif isinstance(v, jax.Array) and do_patch and k == "kernel":
            # KernelShape + (in_features,) + (out_features,)
            new_weights[k] = jnp.concatenate([v * 1e-2, v, v * 1e-3, v * 1e-4], -2)
        elif isinstance(v, jax.Array):
            new_weights[k] = v
        else:
            print(f"Unknown type {type(v)}")
            new_weights[k] = v
    return new_weights


def dict_to_array_dispatch(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return dict_to_array_dispatch(v.item())
        if v.dtype == object:
            raise ValueError(str(v))
        return v
    elif isinstance(v, dict):
        return dict_to_array(v)
    elif isinstance(v, (list, tuple)):
        return list(zip(*sorted(dict_to_array(dict(enumerate(v))).items())))[1]
    else:
        return dict_to_array(v)


def dict_to_array(x):
    new_weights = {}
    for k, v in dict(x).items():
        new_weights[k] = dict_to_array_dispatch(v)
    return new_weights


def load(path: str, prototype: Dict[str, jax.Array]):
    try:
        with smart_open.open(path + ".np", 'rb') as f:
            params = list(zip(*sorted([(int(i), v) for i, v in np.load(f).items()])))[1]
    except:
        with smart_open.open(path + ".np", 'rb') as f:
            params = \
                list(zip(*sorted([(int(i), v) for i, v in np.load(f, allow_pickle=True)["arr_0"].item().items()])))[1]

    _, tree = jax.tree_util.tree_flatten(prototype)
    return tree.unflatten(params)


@app.command()
def main(context: int = 8, base_model: str = "flax/stable-diffusion-2-1", schedule_length: int = 1024,
         t5_tokens: int = 2 ** 13, base_path: str = "gs://video-us/checkpoint/", iterations: int = 4096,
         devices: int = 128):
    global _CONTEXT, _RESHAPE
    _CONTEXT = context
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

    t5_conv = nn.Sequential([nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                             nn.LayerNorm(epsilon=1e-10),
                             nn.relu,
                             nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                             ])

    inp_shape = jax.random.normal(jax.random.PRNGKey(0), (jax.local_device_count(), t5_tokens, 768))
    t5_conv_params = t5_conv.init(jax.random.PRNGKey(0), inp_shape)

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    text_encoder = FlaxLongT5Model.from_pretrained("google/long-t5-tglobal-base", dtype=jnp.float32)
    vae_params = patch_weights(vae_params)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet
    tokenizer: T5Tokenizer = tokenizer
    text_encoder: FlaxLongT5Model = text_encoder

    pos_embd = jax.random.normal(jax.random.PRNGKey(0), (t5_tokens // 64, 1024))
    latent_merge00 = jax.random.normal(jax.random.PRNGKey(0), (1024, 2048))
    latent_merge01 = jax.random.normal(jax.random.PRNGKey(0), (1024, 2048))
    latent_merge01 = latent_merge01 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 1)
    latent_merge1 = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024))
    latent_merge1 = latent_merge1 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 0)
    pos_embd = pos_embd
    external = {"embd": pos_embd, "merge00": latent_merge00, "merge01": latent_merge01, "merge1": latent_merge1}

    vae_params = load(base_path + "vae", vae_params)
    unet_params = load(base_path + "unet", unet_params)
    t5_conv_params = load(base_path + "conv", t5_conv_params)
    external = load(base_path + "embd", external)

    noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    sched_state = noise_scheduler.create_state()

    local_batch = 1
    ctx = context * jax.device_count()

    def get_encoded(input_ids: jax.Array, attention_mask: Optional[jax.Array]):
        input_ids, attention_mask = input_ids.reshape(devices * 8, -1), attention_mask.reshape(devices * 8, -1)
        encoded = text_encoder.encode(input_ids, attention_mask, params=text_encoder.params).last_hidden_state
        encoded = encoded.reshape(devices, -1, 768)  # [batch, t5_tokens, features]
        encoded = t5_conv.apply(t5_conv_params, encoded)
        encoded = encoded.reshape(1, -1, 1024)
        return encoded + external["embd"].reshape(1, -1, 1024)

    def merge(latent, noise):
        shape = noise.shape
        latent = latent.reshape(latent.shape[0], -1) @ external["merge00"]
        noise = noise.reshape(noise.shape[0], -1) @ external["merge01"]
        if latent.shape[0] != noise.shape[0]:
            latent = lax.broadcast_in_dim(latent, (noise.shape[0] // latent.shape[0], *latent.shape), (1, 2))
            latent = latent.reshape(-1, latent.shape[-1])
        out = jnp.maximum(noise + latent, 0) @ external["merge1"]
        return out.reshape(shape)

    def vae_apply(*args, method=vae.__call__, **kwargs):
        global _RESHAPE
        _RESHAPE = True
        out = vae.apply(*args, method=method, **kwargs)
        _RESHAPE = False
        return out

    def sample(input_ids: jax.Array, attention_mask: jax.Array, guidance: int, up, vp):
        tokens = input_ids.size
        unc_tok = jnp.concatenate([jnp.ones((1,)), jnp.zeros((tokens - 1,))]).reshape(input_ids.shape)
        encoded = jnp.concatenate([get_encoded(unc_tok, unc_tok), get_encoded(input_ids, attention_mask)])

        def _step(original_latents, idx):
            inp = lax.dynamic_index_in_dim(original_latents, jnp.maximum(idx - 1, 0), keepdims=True)
            inp = jnp.concatenate([jnp.zeros_like(inp), inp])
            def _inner(state, i):
                latents, state = state
                new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
                pred = unet.apply({"params": up}, merge(inp, new), i, encoded).sample
                uncond, cond = jnp.split(pred, 2, 0)
                pred = uncond + guidance * (cond - uncond)
                return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

            state = noise_scheduler.set_timesteps(sched_state, schedule_length, (1, 4, 16, 16))
            latents = jax.random.normal(jax.random.PRNGKey(idx), (1, 4, 16, 16), jnp.float32)

            (out, _), _ = lax.scan(_inner, (latents, state), jnp.arange(schedule_length)[::-1])
            return original_latents.at[idx].set(out[0]), None

        latents = jnp.zeros((iterations, 4, 16, 16))
        out, _ = lax.scan(_step, latents, jnp.arange(iterations))
        out = jnp.transpose(out, (0, 2, 3, 1)) / 0.18215
        return jnp.transpose(vae_apply({"params": vp}, out, method=vae.decode).sample, (0, 2, 3, 1))

    p_sample = jax.pmap(sample, "batch")

    for i in range(8):
        guidance = 2 ** i  # float(input("Guidance: "))
        inp = """Remember that old PC you stuffed in the closet? Yeah, that one. Today, it's been given a second chance. This old machine isn't worthless, even with it's nine year old CPU and complete lack of a Graphics Card. Why? I like, you know a little secret. Not all PCs need to be for gaming. Tired of paying for Google Drive, host your own Cloud storage. Hate running USB drives through your TV, host your own media server. The possibilities are endless. With the help of Pulseway, the sponsor of today's video, we're going to show you how to repurpose that old rig into your own personal server for cheap or even free. (techno music) The first thing to learn is that servers are just computers. The word describes a role not a specific type of hardware. Sure, the servers in big data centers do look different than your home PC. They're built for reliability with redundant power supplies and ECC Memory. They prefer many slower cores rather than a few fast ones and they lack consumer IO like tons of USB audio or display outputs. But that's because they're serving hundreds, if not thousands of clients. We are not. Which is why even our aging closet PC, an old laptop or even a $30 Raspberry Pi can all act as our first home server. This OptiPlex 7010 we picked up from our local recycler, Free Geek is the perfect candidate, not just because it's cheap at 176 US dollars, but because the bones are solid. It's got a hyper-threading Quad Core Intel i7 3770, a 128 Gig SATA SSD for a boot drive and a reasonable 1 Terabyte of bulk storage. At this price point though, there are some compromises we have to make. It's 12 Gigs of RAM are mismatched and predictably it doesn't have a GPU. And even if we did add one, we would likely run into issues with the power supply because value engineered power supplies like this 250 watt one, while they're generally pretty solid, they don't leave a whole lot of headroom to play with, nor does it offer additional PCI Express Power Cables. But, don't fret. What really matters here is that we have a blank canvas to work from. And by the way, now's a great time to open up your rigging. Give it a good dusting and maybe refresh your thermal paste while you're at it. It needs some love. Now, let's set up our OS. Now before you say it, we're not gonna install Linux today. We know that moving to a Linux or FreeBSD based option does have both performance and feature benefits over Windows. However, if you're just starting out and you've got an old Windows based machine already around, this is the easiest way to get your foot in the door. And that's what we want this video to be. A start, not the end. If you guys wanna see a follow-up where we use TrueNAS or similar, rattle your sabers in the comments and get subscribed. The first thing you're going to want to do is factory reset the PC. Now, if you're running Windows 7 or 8, now is a good time to upgrade to Windows 10 or 11. Keep in mind that a lot of those old Windows keys will still work to activate newer versions of Windows. If you're already there, hit the Windows key and type, Reset. Hah, it's slow. Then click on, Get Started. All you need to do from here is run through the prompts until you have a fresh Windows install. We ran WinAero Tweaker to disable automatic updates and kill Cortana and also disabled lot of the extra bloat like Telemetry that can suck away valuable resources from an older system like this. Now we're ready to set up our server to actually do stuff. For starters, we're going to download Plex Media Server for our media and Pulseway to manage our system remotely. If you're thinking, why would I need remote access to this dinosaur? Well, it's not because we plan on accessing it from across the world but rather because we may not have a spare monitor keyboard and mouse to leave connected to it, let alone the space. Instead, we can operate it headlessly, that is completely controlled via any web browser or the Pulseway app. Neat. Setting up Plex is simple. Install the application, run through the prompts and definitely read that EULA. Am I right? They are very legally binding. The beauty of using a Windows PC to start your server hosting journey is that the user experience isn't, and I mean this with the utmost respect to all command line warriors, complete crap, usually. You just download what you need. The links are in the description and install and configure it with a gooey and then you're done. It really is that easy. We can't cover all the options here, of course. You could run a remote torrenting box or an Ad blocking server, for instance. But if you're following along and you do hit some roadblocks, hop on over to our forum, where there are tons of friendly fellow nerds willing to help you out. Love you guys. Pulseway set up is similarly easy. Just click through the wizard and sign in. At this point, you'll have full access to your files and settings from whatever web browser you use to access Pulseway. And we're done. Service is running, Pulseway is up. You can actually get away with not even setting up a network share or anything and just using Pulseway to manage your files remotely. But having a direct folder share on your local network is pretty nice. So, we're gonna do that too. It just takes a few more steps and you won't be able to access 'em from Starbucks, unfortunately or fortunately. We've got a drive set up here called LTT Simple Server that we'll go into and we have a folder we want to share. What we'll do is we'll right click and go to, Give access to, and then click on specific people. From here, either you can allow only one person to access the file like it is normally or in our case, we're going to give everyone in our local network access. Be sure to change the permissions from Read only to Read and Write, creating a melting pot. And boom, the folder is now able to be added as a Network Share by copying the link into the Add a Network folder prompt in any network machine on your local network. Don't forget to go to the Network and Sharing Center to ensure both Network Discovery and File and Printer Sharing are turned on under the private settings. Or if your network is set as public, you should set that as private because it won't work on public networks, not by default. Double check that. Really, that's it for making a basic file server. And with it, we can do a litany of cool things. We can point our Plex server to the storage folders and access the entire drive over the network. We can set up a Windows remote desktop connection or go easy mode and use Pulseway's built-in Remote Desktop Tool to control our server from anywhere in the world without exposing our IP to the net. We can also update Windows without having to log in to the system itself, even from your phone. And the best part is that we're still running Windows. We're not running Linux. You don't have to learn anything new. If your main rig works itself, you have an easy to configure backup PC, just waiting in the wings. But what if you want more control or what if you don't have a Windows license to throw at your pile of spare parts. You could run a standalone Linux server, of course, but as we've seen in the Linux challenge, that's not always for the faint of heart. Unraid and Proxmox are also feature rich options but they're not super easy to use either. For an intermediate user, we would suggest TrueNAS CORE, previously known as FreeNAS. It's free, easy to install and has a decent UI. That's it, video over. Nah, just kidding. Let's kick it up by adding more storage and data redundancy. These two 12 terabyte IronWolf Pro drives can handle that beautifully. At $400 a pop, they aren't cheap, but considering that 10 terabytes of Google storage costs 70 bucks a month. If we did the math, this upgrade will pay itself off in under a year including the PC, not to mention you will own and control all of your data. All right, we've got the 1 terabyte drive out and the first 12 terabyte IronWolf in, but now we have a problem. Where does this go? We can't exactly like, just throw it here and be done with it. We can't put it here, there's just not enough clearance. Like there's no provisions. There's not even another SATA cable, unless, (Anthony laughs) This optical drive now serves no purpose. Oh, there are some screws in there that I need to get rid of. Ah, okay, cool. But there's another problem. Yes, this freed up some room and yes, this is SATA. Um, they're not the same size. So, that's where this comes in. This will accept a three and a half inch drive, like so and slot in, something like so. Oh, yep, okay. Obviously I need to actually mount that first but that's the plan. Let's get to it. Let's see if it still works. Now with the DVD drive out, because who needs one of these these days, we have a freed up SATA power and data cable right where we need it. So, our second drive, now definitely screwed in is absolutely plugged into the system and I'm not getting a display. This is very troubling. One moment, please. Two things that we learned just now. First, always check your cables. Second, a lot of these old machines sometimes have issues with outputting anything via display port. Yeah, these are the kinds of things that these old systems will sometimes give you trouble with. But, once you have it all set up and running, here we go, we've got 12 terabytes of storage right here. Did I say 12? I meant to say 12 terabytes of redundant storage. Yes. Now it's time to RAID these things. We're gonna be setting up RAID 1, which means the drives will be mirrored. There's no performance benefit on Writes anyway, for Reads, there is. But if one dies, you get to keep 100% of your data. Hit the Windows key and type "Storage spaces". Then create a new pool using the steps we've linked down below. (keys clicking) You will need to format your drives first. Make sure you do that. There we go. On format drives. (Anthony sneezes) Ooh, ah, I'm allergic to Dell (beep). Storage space, Resiliency, two-way mirror, yep. File system, NTFS. Size, so 10.9 terabytes, that's 12 terabytes. And create Storage Space. Formatting the storage space. We are good to go. And there we have it, a 12 terabyte redundant NASBox for under a thousand dollars. And each of these hard drives costs about 400 bucks. So that's telling you something. This particular form factor is actually really easy to hide somewhere in conspicuous, especially if you use Pulseway to do all of your remote monitoring and maintenance. So, huge thanks to Pulseway for sponsoring this video. We hope you enjoyed it. And if you wanna see more of this, sort of DIY server content, don't forget to get subscribed and check out all of our other server videos. We'll have those linked in the end screen for you. For now, see you later.""" # input("Input: ")

        toks = tokenizer(inp, return_tensors="np", padding="max_length", truncation=True, max_length=t5_tokens)
        sample_out = p_sample(np.stack([toks["input_ids"].reshape(devices * jax.local_device_count(), -1)] * 4),
                              np.stack([toks["attention_mask"].reshape(devices * jax.local_device_count(), -1)] * 4),
                              jnp.full((jax.local_device_count(),), guidance, jnp.float32),
                              flax.jax_utils.replicate(unet_params),
                              flax.jax_utils.replicate(vae_params),
)
        Image.fromarray(np.array(sample_out[0]).reshape(-1, 128, 3), "RGB").save(f"out{i}.png")



if __name__ == "__main__":
    app()
