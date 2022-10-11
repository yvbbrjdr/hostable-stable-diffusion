#!/usr/bin/env python3

from transformers import AutoFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

SAFETY_CHECKER = 'CompVis/stable-diffusion-safety-checker'
CLIP_VIT_LARGE = 'openai/clip-vit-large-patch14'

def main():
    AutoFeatureExtractor.from_pretrained(SAFETY_CHECKER)
    StableDiffusionSafetyChecker.from_pretrained(SAFETY_CHECKER)
    CLIPTokenizer.from_pretrained(CLIP_VIT_LARGE)
    CLIPTextModel.from_pretrained(CLIP_VIT_LARGE)

if __name__ == '__main__':
    main()
