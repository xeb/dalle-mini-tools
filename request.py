#!/usr/bin/env python

import fire
from datetime import datetime
from sqs_launcher import SqsLauncher

def send(prompt, queue_name="dalle-mini-tools", region_name="us-east-1"):
    launcher = SqsLauncher(queue_name)

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    safeprompt = prompt.replace(" ", "_").replace("'","").replace('"','').replace("\n","").replace(",", "").lower().strip()
    run_name = f'run_{time}_{safeprompt[:30]}'
    print(f"Using {run_name=}")

    response = launcher.launch_message({ 'prompt': prompt, 'run_name': run_name })
    print(f"Received {response=}")
    return run_name

if __name__ == "__main__":
    fire.Fire(send)
