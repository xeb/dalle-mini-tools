#!/bin/bash
source ./venv/bin/activate
cd dalle_mini_tools
FLASK_APP=server.py FLASK_ENV=development flask run --port 2088
cd ..
