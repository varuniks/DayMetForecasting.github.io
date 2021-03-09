#!/bin/bash
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model lstm --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model lstm 

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model blstm --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model blstm 


python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model ED --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model ED
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model 1dconv --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model 1dconv
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model tcn --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model tcn
#
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model stcn --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000  --model stcn
#
#


python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model lstm --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model lstm 

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model blstm --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model blstm 

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model ED --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model ED

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model 1dconv --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model 1dconv

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model tcn --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model tcn

python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model stcn --train
python3.6 ROM_gen.py --win 7 --modes 5 --epochs 2000 --fb --model stcn

"""

















