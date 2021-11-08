# This is an API file to export an VQVAE/... for tokenization
# Can rewrite the APIs for VQGAN.
# Don't forget to freeze the relavant .py files.

import importlib
import torch
import math
import os

from torchvision.utils import save_image, make_grid
from datetime import datetime

# production APIs

from .vqvae_zc import VQVAE

def new_module(config):
    '''
        in config:
            "target": module type, vqvae_zc.Decoder/vqvae_diffusion.Decoder/vqvae_diffusion.Decoder2
            "ckpt": path of checkpoint
            "ckpt_prefix": prefix to remove in ckpt state dict
            "device": device
            "params": dict of params
    '''
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config.get("target").rsplit(".", 1)
    model = getattr(importlib.import_module(module, package=__package__), cls)(**config.get("params", dict()))

    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    if "ckpt" in config:
        ckpt = torch.load(config.get("ckpt"), map_location='cpu')
        prefix = config.get("ckpt_prefix", None)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]   
        if prefix is not None:
            ckpt = {k[len(prefix) + 1:]: v for k, v in ckpt.items() if k.startswith(prefix)}
        model.load_state_dict(ckpt, strict=False)
        del ckpt
    return model

def load_decoder_default(device=0, path="pretrained/vqvae/l1+ms-ssim+revd_percep.pt"):
    # exp: load currently best decoder
    target = ".vqvae_diffusion.Decoder"
    params = {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [ 1,1,2,4],  # num_down = len(ch_mult)-1
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0
    }
    ckpt_prefix = "dec"

    config = {
        "target": target,
        "params": params,
        "ckpt": path,
        "ckpt_prefix": ckpt_prefix,
        "device": device
    }
    return new_module(config)

# path="/dataset/fd5061f6/cogview/zwd/vqgan/l1+ms-ssim+revd_percep/checkpoints/last.ckpt"
def load_model_default(device=0, 
                    path="pretrained/vqvae/l1+ms-ssim+revd_percep.pt"):
    # exp: load currently best vqvae model
    ddconfig = {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1,1,2,4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0
    }
    params = {
        "in_channel": 3,
        "channel": 512,
        "n_res_block": 0,
        "n_res_channel": 32,
        "embed_dim": 256,
        "n_embed": 8192,
        "stride": 6,
        "simple": True,
        "decay": 0.99,
        "dif": True,
        "ddconfig": ddconfig
    }

    config = {
        'target': ".vqvae_zc.VQVAE",
        'params': params,
        'ckpt': path,
        'device': device
    }
    return new_module(config)

def test_decode(configs, testcase, device=0, output_path=None):
    '''
        configs: list of config for new module
        testcases: pt file path or tensor of [B, D, H, W]
    '''
    if output_path is None:
        output_path = os.path.join("sample", f"{datetime.now().strftime('%m-%d-%H-%M-%S')}.jpg")

    quantize_config = {
        "target": ".vqvae_zc.Quantize",
        "params": {
            "dim": 256,
            "n_embed": 8192,
        },
        "ckpt": "/dataset/fd5061f6/cogview/zwd/pretrained/vqvae/vqvae_hard_biggerset_011.pt",
        "ckpt_prefix": "module.quantize_t",
        "device": device
    }
    quantize = new_module(quantize_config)

    if type(testcase) is str:
        testcase = torch.load(testcase, map_location=torch.device(device))[:, -1024:].contiguous()
        testcase = testcase.view(testcase.shape[0], 32, 32).contiguous()
    else:
        testcase = testcase.to(device)

    quantized_testcase = quantize.embed_code(testcase)
    quantized_testcase = quantized_testcase.permute(0, 3, 1, 2)

    outs = []
    for config in configs:
        decoder = new_module(config)
        out = decoder(quantized_testcase)
        outs.append(out.unsqueeze(0))
    outs = torch.cat(outs).permute(1, 0, 2, 3, 4)
    outs = outs.reshape(-1, *outs.shape[2:]).contiguous()
    save_image(make_grid(outs, nrow=len(configs)), output_path, normalize=True)

def test_decode_default(device=0):
    # testing 3 decoders: original/l1+ms-ssim/l1+ms-ssim+perceptual
    configs = [
        {
            "target": ".vqvae_zc.Decoder",
            "params": {
                "in_channel": 256, 
                "out_channel": 3,
                "channel": 512,
                "n_res_block": 0,
                "n_res_channel": 32,
                "stride": 4,
                "simple": True
            },
            "ckpt": "/dataset/fd5061f6/cogview/zwd/pretrained/vqvae/vqvae_hard_biggerset_011.pt",
            "ckpt_prefix": "module.dec",
            "device": device },
        {
            "target": "vqvae.vqvae_diffusion.Decoder",
            "params": {
                "double_z": False,
                "z_channels": 256,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [ 1,1,2,4],  # num_down = len(ch_mult)-1
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0
            },
            "ckpt": "/dataset/fd5061f6/cogview/zwd/vqgan/l1+ms-ssim/checkpoints/last.ckpt",
            "ckpt_prefix": "dec",
            "device": device },
        {
            "target": "vqvae.vqvae_diffusion.Decoder",
            "params": {
                "double_z": False,
                "z_channels": 256,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [ 1,1,2,4],  # num_down = len(ch_mult)-1
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0
            },
            "ckpt": "/dataset/fd5061f6/cogview/zwd/vqgan/l1+ms-ssim+revd_percep/checkpoints/last.ckpt",
            "ckpt_prefix": "dec",
            "device": device },
    ]
    testcase_dir = "/dataset/fd5061f6/cogview/zwd/vqgan/testcase/"
    for testcase in os.listdir(testcase_dir):
        testcase = os.path.join(testcase_dir, testcase)
        test_decode(configs, testcase, device)

def new_model():
    '''Return a New Instance of VQVAE, the same parameters with the pretrained model.
        This is for torch.load().
    '''
    return VQVAE(
        channel=512, n_res_block=0,
        n_res_channel=32, embed_dim=256,
        n_embed=8192, stride=6
    )

def img2code(model, img):
    '''Convert a batch of img to code
    Args:
        model: The tokenizer model.
        img: [b, c, h, w]
    '''
    with torch.no_grad():
        quant_t1, _, id_t1 = model.encode(img)
    return id_t1.view(img.shape[0], -1) 

def code2img(model, code):
    '''Convert a batch of code to imgs
    Args:
        model: ...
        code: [b, h, w] or [b, h*w] LongTensor
    '''
    if len(code.shape) == 2:
        s = int(math.sqrt(len(code.view(-1))) + 1e-5)
        code = code.view(code.shape[0], s, s)
    with torch.no_grad():
        out = model.decode_code(code)
        out = out * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor([0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1)
    return out

