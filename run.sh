#!/bin/bash

for (( i=1; i<=18; i++ ))
do
#     ./test image/l$i.png right.jpeg
    for (( j=1; j<=18; j++ ))
    do
#        echo l$i.png r$j.png
        ./test image/r$i.png image/l$j.png
    done
done
