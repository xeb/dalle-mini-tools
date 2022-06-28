#!/usr/bin/env python

import os
import subprocess

import fire
from sqs_listener import SqsListener

from generate import Generator


class ImgGenListener(SqsListener):
    def init_model(self, output_dir, clip_scores, postprocess, postprocess_cwd):
        self.generator = Generator(output_dir, clip_scores)
        self.postprocess = postprocess
        self.postprocess_cwd = postprocess_cwd
        print("Initialized model")

    def postprocessing(self, run_name):
        if not self.postprocess or len(self.postprocess) == 0:
            print("Postprocessing not enabled, skipping...")
            return

        if os.path.exists(self.postprocess):
            cmds = [f"{self.postprocess}", run_name]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.postprocess_cwd)
            out, err = p.communicate()
            if p.returncode != 0:
                print(f"Exception\n{err=}\n\n{out=}")
            print(f"Postprocess complete for {run_name=}")
        else:
            print(f"Skipping postprocessing as {self.postprocess=} does not exist")

    def handle_message(self, body, attr, msg_attr):
        print(f"Processing {body=} {attr=} {msg_attr=}")
        prompt = body["prompt"]
        run_name = body["run_name"]
        self.generator.generate(prompt=prompt, run_name=run_name)
        self.postprocessing(run_name)
        print(f"Processed! {body=}")


def main(
    output_dir="output",
    clip_scores=False,
    postprocess="",
    postprocess_cwd="",
    queue_name="dalle-mini-tools",
    error_queue="dalle-mini-tools_errors",
    region_name="us-east-1",
    interval=1,
):

    print(f"Running worker with {queue_name=}")
    listener = ImgGenListener(
        queue_name, error_queue=error_queue, region_name=region_name, interval=interval
    )

    listener.init_model(output_dir, clip_scores, postprocess, postprocess_cwd)
    listener.listen()


if __name__ == "__main__":
    fire.Fire(main)
