#!/bin/bash

echo $1

export WORKON_HOME=~/Envs
source /usr/local/bin/virtualenvwrapper.sh
workon env1

nohup python train.py --config $1 --gpu 0 > logs/$1-$(date +%F-%T).txt 2>&1 &