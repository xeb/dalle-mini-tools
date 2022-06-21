#!/usr/bin/env python

import os
import fire
import glob
import jinja2
from pathlib import Path

def generate_all(output_dir="output"):
    print(f"Processing {output_dir}")
    for path in Path(output_dir).iterdir():
        print(f"Processing {path}")
        if path.is_dir():
            index_content = generate_index(path)
            with open(f"{path}/index.html", "w") as o:
                o.write(index_content)
            print(f"Wrote {path=}/index.html")

def get_dir_details(path):
    prompt_path = f"{path}/prompt.txt"
    if os.path.exists(prompt_path) is False:
        print("Skipping {path=} because {prompt_path=} does not exist")
        return None, None

    prompt = ""
    with open(f"{path}/prompt.txt", "r") as rp:
        prompt = rp.read()

    imgs = [ os.path.basename(x) for x in glob.glob(f'{path}/*.png') ]
    return ( prompt, imgs )

def generate_index(path):
    tl = jinja2.FileSystemLoader(searchpath="./templates")
    te = jinja2.Environment(loader=tl)
    template = te.get_template("template.html")
    prompt, imgs = get_dir_details(path)
    return template.render(prompt=prompt, imgs=imgs)

if __name__ == "__main__":
    fire.Fire(generate_all)
