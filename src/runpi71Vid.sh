#!/bin/sh

#ms of time duration
LENGTH=1000

raspivid -t $LENGTH -w 1640 -h 1232 --framerate 25 -o pi71_`date +%s`.h264


