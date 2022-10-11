#!/usr/bin/env python3

import argparse
import os
import random

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
import torch
from transformers import AutoFeatureExtractor

from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

SAFETY_MODEL_ID = 'CompVis/stable-diffusion-safety-checker'

def load_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()
    return model

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype(np.uint8)
    return [Image.fromarray(image) for image in images]

def load_replacement(x, rick_path):
    h, w, _ = x.shape
    y = Image.open(rick_path).convert('RGB').resize((w, h))
    return (np.array(y) / 255).astype(x.dtype)

def check_safety(samples, safety_feature_extractor, safety_checker, rick_path):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(samples), return_tensors='pt').pixel_values
    samples, has_nsfw_concept = safety_checker(images=samples, clip_input=safety_checker_input)
    for i, sample in enumerate(samples):
        if has_nsfw_concept[i]:
            samples[i] = load_replacement(sample, rick_path)
    return samples

def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_everything(args.seed)

    app_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    stable_diffusion_path = os.path.join(app_root, 'src/vendor/stable-diffusion')
    config_path = os.path.join(stable_diffusion_path, 'configs/stable-diffusion/v1-inference.yaml')
    checkpoint_path = os.path.join(stable_diffusion_path, 'models/ldm/stable-diffusion-v1/model.ckpt')
    rick_path = os.path.join(stable_diffusion_path, 'assets/rick.jpeg')

    model = load_model(config_path, checkpoint_path)
    sampler = PLMSSampler(model)
    prompts = [args.prompt] * args.batch_size
    empty_prompts = [''] * args.batch_size

    if not args.disable_nsfw_filter:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(SAFETY_MODEL_ID)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(SAFETY_MODEL_ID)

    with torch.no_grad(), torch.autocast('cuda'), model.ema_scope():
        count = 0
        for _ in range(args.n_batches):
            samples = sampler.sample(
                S=args.sampling_steps,
                conditioning=model.get_learned_conditioning(prompts),
                batch_size=args.batch_size,
                shape=[4, args.height // 8, args.width // 8],
                verbose=False,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=model.get_learned_conditioning(empty_prompts)
            )[0]
            samples = model.decode_first_stage(samples)
            samples = torch.clamp((samples + 1) / 2, min=0, max=1)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            if not args.disable_nsfw_filter:
                samples = check_safety(samples, safety_feature_extractor, safety_checker, rick_path)
            samples = (samples * 255).round().astype(np.uint8)

            for sample in samples:
                img = Image.fromarray(sample)
                img.save(os.path.join(args.outdir, f'{count:03}.png'))
                count += 1

def parse_args():
    parser = argparse.ArgumentParser('Text to image')
    parser.add_argument('prompt', type=str, help='Prompt to use for the image')
    parser.add_argument('-o', '--outdir', type=str, default='output', help='Output directory')
    parser.add_argument('-r', '--resolution', type=str, default='512', help='Resolution of the image')
    parser.add_argument('-n', '--n-batches', type=int, default=1, help='Number of batches')
    parser.add_argument('-bs', '--batch-size', type=int, default=9, help='Batch size')
    parser.add_argument('-ss', '--sampling-steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('-gs', '--guidance-scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Seed for the random number generator')
    parser.add_argument('-dnf', '--disable-nsfw-filter', action='store_true', help='Disable the NSFW filter')

    args = parser.parse_args()

    if args.prompt == '':
        raise ValueError('Prompt cannot be empty')

    resolution = tuple(map(int, args.resolution.split('x')))
    if len(resolution) > 2:
        raise ValueError('Invalid resolution format. Must be in the format WxH or L')
    if len(resolution) == 1:
        args.width = args.height = resolution[0]
    else:
        args.width, args.height = resolution
    if args.width <= 0 or args.height <= 0:
        raise ValueError('Resolution must be positive.')
    if args.width % 64 != 0 or args.height % 64 != 0:
        raise ValueError('Resolution must be divisible by 64.')

    if args.n_batches <= 0:
        raise ValueError('Number of batches must be positive.')
    if args.batch_size <= 0:
        raise ValueError('Batch size must be positive.')
    if args.sampling_steps <= 0:
        raise ValueError('Sampling steps must be positive.')
    if args.guidance_scale < 0:
        raise ValueError('Guidance scale must be non-negative.')

    if args.seed < 0:
        args.seed = random.randint(0, 1000000)

    return args

if __name__ == '__main__':
    main()
