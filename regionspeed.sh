#!/bin/bash
file=logs/regionspeed.txt
echo $file
nohup python regionspeed.py > $file  2>&1 &
tail -f $file -n 100
