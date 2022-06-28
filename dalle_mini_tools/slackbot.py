#!/usr/bin/env python

import os
import re
import fire
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from tqdm import tqdm

from request import send as send_queue_request

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OUTPUT_DIR = "./output"
BASE_URI = "https://example.com/output"

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def mention_handler_app_mention(body, say, logger):
    global OUTPUT_DIR
    global BASE_URI

    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    channel = event.get("channel", None) or event["channel"]

    print(f"Handling {thread_ts}")
    prompt = (
        event["text"].replace(event["text"].split(" ")[0].strip(), "").strip()
    )  # i feel dirty but its late
    
    app.client.reactions_add(channel=channel, timestamp=thread_ts, name="thumbsup")
    print(f"Generating {prompt=}")
    rundir = send_queue_request(prompt)

    max_t = 16000000  # my 2080Ti can generate from SQS to final image in: 1672800 ticks
    for i in tqdm(range(max_t)):
        if i % 100000 == 0:
            print(f"Checking {rundir} {i}/{max_t}")

        if os.path.exists(f"{OUTPUT_DIR}/{rundir}/final.png"):
            img = f"{BASE_URI}/{rundir}/final.png"
            print(f"Found {img}")
            logger.info(f"Found {img}")
            say(img)
            return

    say(text="...I gave up.", thread_ts=thread_ts)


@app.event("message")
def mention_handler_message(body, say):
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    messag_ts = event.get("ts", None)
    if "text" not in event:
        return

    channel = event.get("channel", None) or event["channel"]
    message = event["text"].strip()

    if message in [str(x) for x in range(0,9)]:
        print(f"Processing {message}")
        idx = int(message) - 1 # ask for 0, get 0, ask for 1, get 0
        print(f"{idx=}")
        if idx > 8:
            idx = 8
        if idx < 0:
            idx = 0
        print(f"{idx=}")

        # ACK the request from the original event timestamp
        app.client.reactions_add(channel=channel, timestamp=messag_ts, name="thumbsup")

        replies = app.client.conversations_replies(channel=channel, ts=thread_ts)
        root = replies["messages"][0]["text"]
        print(f"Getting img {idx} from {root}")
        p = r"^.*(run_.*)/.*"
        m = re.search(p, root)
        if m:
            run = m.group(1)
            img = root.replace("final.png", f"img_{idx}.png")
            print(f"Returning {img=}")
            say(img, thread_ts=thread_ts)

    if "generation station" in message.lower():
        say(
            text=(
                "What was that? Did you want to generate an image? Just mention me"
                " (@ImageGen) and tell me what you want."
            ),
            thread_ts=thread_ts,
        )

def main(output_dir="", base_uri=""):
    global OUTPUT_DIR
    global BASE_URI

    if output_dir != "":
        OUTPUT_DIR = output_dir
    
    if base_uri != "":
        BASE_URI = base_uri

    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
    fire.Fire(main)
