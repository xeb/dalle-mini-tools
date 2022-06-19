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

I'm a fan of CLIs and getting things running right away with minimal code. The original dalle-mini repo is great, but I wanted to share something quick & easy for those who just want to use the model.