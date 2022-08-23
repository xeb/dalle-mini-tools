import fire
import torch
from pathlib import Path
from torch import autocast
from slugify import slugify
from datetime import datetime
from os.path import expanduser
from diffusers import StableDiffusionPipeline


class StableDiffusionGenerator:
    def __init__(self, output_dir="output", clip_scores=False):
        self.output_dir = output_dir

        token = ""
        with open(expanduser("~/.huggingface/token"), "r") as f:
            token = f.read()
    
        def unsafe(images, clip_input):
            """ Don't judge me, future"""
            return images, False

        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=token)
        self.model.safety_checker = unsafe
        self.model = self.model.to("cuda")


    def generate(self, prompt, clip_scores=False, run_name=None):

        image = None

        with autocast("cuda"):
            image = self.model(prompt)["sample"][0]
        
        safeprompt = prompt

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
            f.write(prompt)

        path = f"{output_dir_}/img_0.png"
        image.save(path)
        print(f"Saved to {path=}")


def main(prompt, output_dir="output", clip_scores=False):
    generator = StableDiffusionGenerator(output_dir=output_dir, clip_scores=clip_scores)
    dir = generator.generate(prompt)
    return dir


if __name__ == "__main__":
    fire.Fire(main)
