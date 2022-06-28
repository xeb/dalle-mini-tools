#!/bin/bash
RUN_NAME=$1
cd output/$1
HEIGHT=128
convert -size 1024x$HEIGHT -background white -bordercolor white -pointsize 32 -fill black caption:"$(cat prompt.txt)" -flatten -border 10x15 title.png
number=1
for f in img_*.png; do
    convert "$f" -gravity southwest -background gray90 -pointsize 18 -weight heavy label:"$number" -composite miff:-
    ((number++))
done | montage -geometry +2+2 -bordercolor white -border 10x0 - all.png
montage title.png all.png -geometry +1+1 -gravity south -tile 1x final.png
rm title.png
rm all.png
cd ../..
