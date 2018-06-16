#!/bin/bash

echo $1
file=logs/$1-$(date +%F-%T).txt
echo $file
nohup python train.py --config $1 --gpu 3 > $file  2>&1 &
tail -f $file -n 100
