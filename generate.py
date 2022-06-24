#!/usr/bin/env python

import random
from datetime import datetime
from functools import partial
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key
from PIL import Image
from slugify import slugify
from tqdm.notebook import trange
from transformers import CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel


class Generator:
    def __init__(self, output_dir="output", clip_scores=False):
        self.output_dir = output_dir
        self.clip_scores = clip_scores

        self.DALLE_COMMIT_ID = None

        # if the notebook crashes too often you can use dalle-mini instead by
        # uncommenting below line

        # can be wandb artifact or 🤗 Hub or local folder or google bucket
        # self.DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
        self.DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"

        # VQGAN model
        # https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384
        self.VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        self.VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

        # check how many devices are available
        self.dc = jax.local_device_count()
        print(f"Found {self.dc} devices")

        # Load dalle-mini
        self.model, self.params = DalleBart.from_pretrained(
            self.DALLE_MODEL,
            revision=self.DALLE_COMMIT_ID,
            dtype=jnp.float16,
            _do_init=False,
        )

        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self.VQGAN_REPO, revision=self.VQGAN_COMMIT_ID, _do_init=False
        )

        self.params = replicate(self.params)
        self.vqgan_params = replicate(self.vqgan_params)

        # create a random key
        self.seed = random.randint(0, 2**32 - 1)
        self.key = jax.random.PRNGKey(self.seed)

        print("Creating processor...")
        self.processor = DalleBartProcessor.from_pretrained(
            self.DALLE_MODEL, revision=self.DALLE_COMMIT_ID
        )

        # number of predictions per prompt
        self.n_predictions = 8

        # We can customize generation parameters
        # (see https://huggingface.co/blog/how-to-generate)
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0

    def generate(self, prompt, clip_scores=False, run_name=None):

        # create these partials within the generation runtime
        # model inference
        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
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
            return self.vqgan.decode_code(indices, params=params)

        print(f"Generating image for {prompt}")
        prompts = [prompt]

        print("Processing prompts...")
        tokenized_prompts = self.processor(prompts)
        tokenized_prompt = replicate(tokenized_prompts)

        print(f"Prompts: {prompts}\n")
        # generate images
        images = []
        safeprompt = ""
        for prompt in prompts:
            safeprompt = safeprompt + prompt

        if run_name is None:
            time = datetime.now().strftime("%Y%m%d-%H%M%S")
            safeprompt = slugify(
                prompt, max_length=30, allow_unicode=False, word_boundary=True
            )
            run_name = f"run_{time}_{safeprompt}"
            print(f"Using {run_name=}")

        output_dir_ = f"{self.output_dir}/{run_name}"
        Path(output_dir_).mkdir(parents=True, exist_ok=True)

        with open(f"{output_dir_}/prompt.txt", "w") as f:
            for prompt in prompts:
                f.write(prompt)

        for i in trange(max(self.n_predictions // jax.device_count(), 1)):
            # get a new key
            self.key, self.subkey = jax.random.split(self.key)
            # generate images
            print(f"Generating image {i}...")
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(self.subkey),
                self.params,
                self.gen_top_k,
                self.gen_top_p,
                self.temperature,
                self.cond_scale,
            )

            Path(output_dir_).mkdir(parents=True, exist_ok=True)

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # decode images
            decoded_images = p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))

            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
                path = f"{output_dir_}/img_{i}.png"
                img.save(path)
                print(f"Saved to {path=}")

        if clip_scores:
            # CLIP model
            CLIP_REPO = "openai/clip-vit-base-patch32"
            CLIP_COMMIT_ID = None

            # Load CLIP
            clip, clip_params = FlaxCLIPModel.from_pretrained(
                CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
            )
            clip_processor = CLIPProcessor.from_pretrained(
                CLIP_REPO, revision=CLIP_COMMIT_ID
            )
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
            #         display(images[idx * p + i])
            #         print(
            #             "Score:"
            #             + f" {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n"
            #         )
            #     print()

        return output_dir_


def main(prompt, output_dir="output", clip_scores=False):
    generator = Generator(output_dir=output_dir, clip_scores=clip_scores)
    dir = generator.generate(prompt)
    return dir


if __name__ == "__main__":
    fire.Fire(main)
