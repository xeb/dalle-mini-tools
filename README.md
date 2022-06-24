# dalle-mini-tools

A (soon-to-be) collection of tools for generating [dalle-mini](https://github.com/borisdayma/dalle-mini) images

## Installation & Usage

Install the dependencies, then try out the CLI. Try `python generate.py --help` for more.

```sh
# If you installed poetry 1.x before, uninstall first
curl -sSL https://install.python-poetry.org | python3 - --uninstall

# Install poetry 1.2.x
curl -sSL https://install.python-poetry.org | python3 - --preview

# Create virtual env for this project, install requirements
poetry install

# Enter virtual environment
poetry shell

# Generate an image from text
python generate.py "a man at a computer trying to generate images"

# Alternatively, rather than enter the venv with `poetry shell`, run directly:
poetry run python generate.py --help
```

and if everything runs OK, you should get images in an `output` directory. Like:

![dalle-mini Samples](assets/dalle-mini-samples.png)

## Purpose

This is repository is a collection of tools for doing inference against dalle-mini and dalle-mega.

## What is in this project?

To date, the project contains:

* __`generate.py`__ is a command-line interface for generating images. This has no dependencies.
* __`sitegen.py`__ is a static website generator that uses `templates/template.html` to create index pages per the specified `output_dir` so you can upload results to a webserver
* __`server.py`__ is a Flask web server to host requests. This depends the `request.py` library and on having a `worker.py` running since requests are queued into an SQS queue.
* __`worker.py`__ is a worker process that listens to a SQS queue and then runs the model (via `generate.py`)
* __`request.py`__ is a command-line tool (and library used by `server.py`) for sending requests to the SQS queue. The server depends on this.

## Development notes

- Autoformat python files with `poetry run make lint`
- Installs `ipywidgets` to avoid a tqdm error: `AttributeError: 'tqdm_notebook' object has no attribute 'disp'`
- Installs `tokenizers` 0.11.6 to support running on Apple silicon (0.12.x isn't working)
- You may need to `pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` to get NVIDIA GPU support; will see how to do this in Poetry later

More to come...
