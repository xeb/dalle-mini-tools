#!/bin/bash
RUN_NAME=$1
cd output/$1
HEIGHT=128
convert -size 1024x$HEIGHT -background white -bordercolor white -pointsize 32 -fill black caption:"$(cat prompt.txt)" -flatten -border 10x15 title.png
number=1
for f in img_*.png; do
    convert \( -size 18x18 xc:white \) \
    \( +clone  -alpha extract \
    -draw 'fill black polygon 0,0 0,15 15,0 fill white circle 1,1 15,0' \
    \( +clone -flip \) -compose Multiply -composite \
    \( +clone -flop \) -compose Multiply -composite \
    \) -alpha off -compose CopyOpacity -composite \
    \( -size 18x18 -background none -fill white -gravity center \
    -pointsize 18 -family arial -weight Heavy label:"$number" \) \
    -gravity southwest -compose dstout -composite -alpha on \
    "$f" +swap -compose over -composite miff:-
    ((number++))
done | montage -geometry +2+2 -bordercolor white -border 10x0 - all.png
montage title.png all.png -geometry +1+1 -gravity south -tile 1x final.png
rm title.png
rm all.png
cd ../..
