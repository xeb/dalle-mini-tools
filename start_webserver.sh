#!/bin/bash
source ./venv/bin/activate
FLASK_APP=dalle_mini_tools/webserver.py FLASK_ENV=development flask run --port 2088
