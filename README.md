# dalle-mini-tools
A (soon-to-be) collection of tools for generating [dalle-mini](https://github.com/borisdayma/dalle-mini) images 

# Installation & Usage
Install the dependencies, then try out the basebones CLI. Try `python cli.py --help` for options like saving a wandb run.

```
pip install -r requirements.txt
python cli.py "a man at a computer trying to generate images"
```

and if everything runs OK, you should get images in an `output` directory. Like:
<br/><br/>

![dalle-mini Samples](assets/dalle-mini-samples.png)

# Why does this repository exist?

I'm a fan of CLIs, (and web interfaces, and chat bots, and Alexa skills) and getting things running right away with minimal code. The original dalle-mini repo is great, but I wanted to share something quick & easy for those who just want to use the model through various interfaces. This is all intended for my own synthetic image generation research.


# What is in this project?
To date, the project contains:

* __`cli.py`__ as a command-line interface for generating images
* __`sitegen.py`__ as a static website generator that uses `template.html` to create a quick website per the specified `output_dir` so you can upload results to a webserver
* __`server.py`__ as a server to host the dalle-mini model in a Flask server, super basic.

More to come...

