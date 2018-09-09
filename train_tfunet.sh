#!/bin/bash
file=logs/tfunet-$(date +%F-%T).txt
echo $file
nohup python train_tfunet.py > $file  2>&1 &
tail -f $file -n 100
