#!/bin/sh
#$-cwd
#$-j y
#$-V
#$-o ./log/search.log
#$-pe gpu 1
#$-l h=yagi11,gpu=1
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/anhcda/ANAS/anas/search_net.py \
--batch_size 128 \
--seed 1 \
--note exp1 \
>> /home/anhcda/ANAS/anas/outputs/search_output.txt
