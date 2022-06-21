#!/usr/bin/env python

import os
import fire
import uuid
import jax
import jax.numpy as jnp
from pathlib import Path
from dalle_mini import DalleBart, DalleBartProcessor
import wandb
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from functools import partial
import random
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
from flax.training.common_utils import shard

def main(prompt, output_dir="output", clip_scores=False):
    print(f"Generating image for {prompt}")
    prompts = [ prompt ]        

    #DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
    DALLE_COMMIT_ID = None

    # if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
    DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"

    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    # check how many devices are available
    dc = jax.local_device_count()
    print(f"Found {dc} devices")

    # Load dalle-mini
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )

    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    # number of predictions per prompt
    n_predictions = 8

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    print(f"Prompts: {prompts}\n")
    # generate images
    images = []
    safeprompt = ""
    for prompt in prompts:
        safeprompt = safeprompt + prompt

    safeprompt = safeprompt.replace(" ", "_").replace(",", "").lower().strip()
    output_dir_ = f'{output_dir}/run_{str(uuid.uuid4())[:5]}_{safeprompt[:20]}'
    Path(output_dir_).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir_}/prompt.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt)

    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        
        Path(output_dir_).mkdir(parents=True, exist_ok=True)
        
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            path = f'{output_dir_}/img_{i}.png'
            img.save(path)
            print(f'Saved to {path=}')

    if clip_scores:
        # CLIP model
        CLIP_REPO = "openai/clip-vit-base-patch32"
        CLIP_COMMIT_ID = None

        # Load CLIP
        clip, clip_params = FlaxCLIPModel.from_pretrained(
            CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
        clip_params = replicate(clip_params)

        # score images
        @partial(jax.pmap, axis_name="batch")
        def p_clip(inputs, params):
            logits = clip(params=params, **inputs).logits_per_image
            return logits

        # get clip scores
        clip_inputs = clip_processor(
            text=prompts * jax.device_count(),
            images=images,
            return_tensors="np",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).data
        logits = p_clip(shard(clip_inputs), clip_params)

        # organize scores per prompt
        p = len(prompts)
        logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()

        # for i, prompt in enumerate(prompts):
        #     print(f"Prompt: {prompt}\n")
        #     for idx in logits[i].argsort()[::-1]:
        #         #display(images[idx * p + i])
        #         # print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
        #     print()


    return output_dir_

if __name__ == "__main__":
    fire.Fire(main)
