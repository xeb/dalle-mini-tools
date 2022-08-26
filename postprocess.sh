#!/bin/bash
# This is a postprocessing script. For now we just create a montage for the run
# In the future I plan to use ESRGAN for upscaling
# cp output/$1/img_0.png output/$1/final.png

# The montage was used for dalle-mini, but switching to Stable Diffusions
./montage.sh $1
