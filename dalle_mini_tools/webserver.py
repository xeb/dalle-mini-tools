#!/usr/bin/env python

import os
import time

import fire
from flask import Flask, redirect, render_template, request, send_from_directory

from request import send as send_queue_request
from sitegen import get_dir_details

flaskapp = Flask("dalle-mini")

# TODO: make output a command line argument

@flaskapp.route("/", methods=["POST", "GET"])
def root():
    if request.method == "GET":
        return render_template("request.html")
    elif request.method == "POST":
        prompt = request.form.get("prompt")
        print(f"processing {prompt}")
        prompt = prompt.replace(",", " ")
        rundir = send_queue_request(prompt)
        return redirect(
            f"/output/{rundir}/index.html"
        )  # TODO: should output be configurable?


@flaskapp.route("/output/<path:path>")
def output(path):
    print(f"Handling {path=}")
    if os.path.basename(path) == "index.html":
        run_name = os.path.dirname(path).replace(os.path.basename(path), "")
        print(f"Handling {run_name=}")
        ddir = os.path.join("./output", os.path.dirname(path))

        expectedimgs = 1 # only doing 1 for Stable Diffusion for now

        if not os.path.exists(ddir):
            # Not generated yet, keep waiting
            time.sleep(1)
            return render_template("refresh.html", run_name=run_name)

        print(f"Processing {ddir}")
        prompt, imgs = get_dir_details(ddir)
        if imgs is None or len(imgs) == 0:
            print(f"No images found for request path {path}")
            time.sleep(1)
            return redirect(f"/output/{path}")

        return render_template(
            "template.html",
            prompt=prompt,
            imgs=imgs,
            expected_img_count=expectedimgs,
            show_links=True,
        )

    else:
        return send_from_directory("./output", path)

def main(port=2088):
    flaskapp.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    fire.Fire(main)
