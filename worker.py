#!/usr/bin/env python

import fire
from sqs_listener import SqsListener

from generate import Generator


class ImgGenListener(SqsListener):
    def init_model(self, output_dir, clip_scores):
        self.generator = Generator(output_dir, clip_scores)

    def handle_message(self, body, attr, msg_attr):
        print(f"Processing {body=} {attr=} {msg_attr=}")
        prompt = body["prompt"]
        run_name = body["run_name"]
        self.generator.generate(prompt=prompt, run_name=run_name)
        print(f"Processed! {body=}")


def main(
    output_dir="output",
    clip_scores=False,
    queue_name="dalle-mini-tools",
    error_queue="dalle-mini-tools_errors",
    region_name="us-east-1",
    interval=1,
):

    print(f"Running worker with {queue_name=}")
    listener = ImgGenListener(
        queue_name, error_queue=error_queue, region_name=region_name, interval=interval
    )

    listener.init_model(output_dir, clip_scores)
    listener.listen()


if __name__ == "__main__":
    fire.Fire(main)
