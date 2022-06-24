#!/usr/bin/env python

from datetime import datetime

import fire
from slugify import slugify
from sqs_launcher import SqsLauncher


def send(prompt, queue_name="dalle-mini-tools", region_name="us-east-1"):
    launcher = SqsLauncher(queue_name)

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    safeprompt = slugify(prompt, max_length=30, allow_unicode=False, word_boundary=True)
    run_name = f"run_{time}_{safeprompt}"
    print(f"Using {run_name=}")

    response = launcher.launch_message({"prompt": prompt, "run_name": run_name})
    print(f"Received {response=}")
    return run_name


if __name__ == "__main__":
    fire.Fire(send)
