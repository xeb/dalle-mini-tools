#!/usr/bin/env python

import os

from django.shortcuts import render
import cli
import fire
from sitegen import get_dir_details
from flask import Flask, request, render_template, redirect, send_from_directory, has_request_context

flaskapp = Flask("dalle-mini")

@flaskapp.route("/", methods=['POST', 'GET'])
def root():
    if request.method == 'GET':
        return render_template('request.html')
    elif request.method == 'POST':
        prompt = request.form.get('prompt')
        print(f"processing {prompt}")
        prompt = prompt.replace(",", " ")
        rundir = cli.main(prompt)
        return redirect(f'/{rundir}/index.html')


@flaskapp.route("/output/<path:path>")
def output(path):
    print(f"Handling {path=}")
    if os.path.basename(path) == "index.html":
        ddir = os.path.join('output', os.path.dirname(path))
        print(f'Processing {ddir}')
        prompt, imgs = get_dir_details(ddir)
        return render_template("template.html", prompt=prompt, imgs=imgs)
    else:
        return send_from_directory('output', path)

def main(port=2088):
    flaskapp.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    fire.Fire(main)