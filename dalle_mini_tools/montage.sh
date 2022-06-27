#!/bin/bash
RUN_NAME=$1
cd output/$1
HEIGHT=128
convert -size 1024x$HEIGHT -background white -bordercolor white -pointsize 32 -fill black caption:"$(cat prompt.txt)" -flatten -border 10x15 title.png
montage img_0.png img_1.png img_2.png img_3.png img_4.png img_5.png img_6.png img_7.png -geometry +2+2 -bordercolor white -border 10x0 all.png
montage title.png all.png -geometry +1+1 -gravity south -tile 1x final.png
rm title.png
rm all.png
cd ../..
