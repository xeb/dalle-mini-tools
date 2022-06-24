#!/usr/bin/env python

import os
import time
from tqdm import tqdm
from request import send as send_queue_request
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
SLACK_APP_TOKEN = os.environ['SLACK_APP_TOKEN']

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def mention_handler(body, say, logger):
	event = body["event"]
	thread_ts = event.get("thread_ts", None) or event["ts"]
	prompt = event["text"].replace(event["text"].split(' ')[0].strip(), "").strip() # i feel dirty but its late
	print(f"Generating {prompt=}")
	# start = time.time()
	rundir = send_queue_request(prompt)
	say(text=f'On it!', thread_ts=thread_ts)

	max_t = 8000000 # my 2080Ti can generate from SQS to final image in: 1672800 ticks
	for i in tqdm(range(max_t)):
		if i % 100000 == 0:
			print(f"Checking {rundir} {i}/{max_t}")
		
		if os.path.exists(f'output/{rundir}/final.png'):
			img = f'https://dalle-mini-tools.xeb.ai/output/{rundir}/final.png'
			print(f"Found {img}")
			say(img)
			# say(img, thread_ts=thread_ts)
			# end = time.time()
			# duration = (start - end)
			# # if duration >= 86400:
			# # 	days = int(duration / 86400)
			# elapsed = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(duration))
			# say(f'Took me {elapsed}.', thread_ts=thread_ts)
			return

	say(text=f'...I gave up.', thread_ts=thread_ts)

@app.event("message")
def mention_handler(body, say):
	event = body["event"]
	thread_ts = event.get("thread_ts", None) or event["ts"]
	if "text" not in event:
		return

	message = event["text"].strip()
	if "generation station" in message.lower():
		say(text='What was that? Did you want to generate an image? Just mention me (@ImageGen) and tell me what you want.', thread_ts=thread_ts)

if __name__ == "__main__":
	handler = SocketModeHandler(app, SLACK_APP_TOKEN)
	handler.start()
