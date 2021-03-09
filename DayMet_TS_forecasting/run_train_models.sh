#!/bin/bash
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model lstm --train

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model blstm --train

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model ED --train
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model 1dconv --train
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model tcn --train
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model stcn --train
#

















