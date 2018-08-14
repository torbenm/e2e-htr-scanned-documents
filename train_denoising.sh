#!/bin/bash
file=logs/dncnn-$(date +%F-%T).txt
echo $file
nohup python train_denoising.py --gpu 3 ${@:1} > $file  2>&1 &
tail -f $file -n 100
