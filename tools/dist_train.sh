#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
<<<<<<< HEAD
=======

# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py ./configs/gfocal_maedet/maedet_l_6x_lr0.02.py  --launcher pytorch
>>>>>>> origin/master
