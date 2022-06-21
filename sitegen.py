#!/usr/bin/env python

import os
import fire
from pathlib import Path

def main(output_dir="output"):
    idx = ""
    with open("template.html", "r") as i:
        idx = i.read()

    print(f"Processing {output_dir}")
    for path in Path(output_dir).iterdir():
        print(f"Processing {path}")
        if path.is_dir():
            prompt_path = f"{path}/prompt.txt"
            if os.path.exists(prompt_path) is False:
                print("Skipping {path=} because {prompt_path=} does not exist")
                continue

            prompt = ""
            with open(f"{path}/prompt.txt", "r") as rp:
                prompt = rp.read()

            with open(f"{path}/index.html", "w") as o:
                o.write(idx.replace("{prompt}", prompt))

            print(f"Wrote {path=} with {prompt=} to index")

if __name__ == "__main__":
    fire.Fire(main)
