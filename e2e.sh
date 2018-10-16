#!/bin/bash
echo $1
file=logs/e2e-$1-$(date +%F-%T).txt
echo $file
nohup python e2e.py --config $1 > $file  2>&1 &
tail -f $file -n 100
