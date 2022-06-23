#!/usr/bin/env python

import os
import fire
import subprocess
from generate import Generator
from sqs_listener import SqsListener

class ImgGenListener(SqsListener):
    def init_model(self, output_dir, clip_scores, postprocess):
        self.generator = Generator(output_dir, clip_scores)
        self.postprocess = postprocess

    def postprocessing(self, run_name):
        if not self.postprocess:
            print(f"Postprocessing not enabled, skipping...")

        if os.path.exists("postprocess.sh"):
            cmds = [ "./postprocess.sh", run_name ]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if p.returncode != 0:
                print("-"**5)
                print(f"Exception\n{err=}\n\n{out=}")
            

    def handle_message(self, body, attr, msg_attr):
        print(f"Processing {body=} {attr=} {msg_attr=}")
        prompt = body["prompt"]
        run_name = body["run_name"]
        self.generator.generate(prompt=prompt, run_name=run_name)
        self.postprocessing(run_name)
        print(f"Processed! {body=}")

def main(output_dir="output", 
    clip_scores=False, 
    postprocess=True,
    queue_name="dalle-mini-tools", 
    error_queue="dalle-mini-tools_errors", 
    region_name="us-east-1",
    interval=1):
    
    print(f"Running worker with {queue_name=}")
    listener = ImgGenListener(queue_name, 
        error_queue=error_queue, 
        region_name=region_name, 
        interval=interval)

    listener.init_model(output_dir, clip_scores, postprocess)
    listener.listen()

if __name__ == "__main__":
    fire.Fire(main)
