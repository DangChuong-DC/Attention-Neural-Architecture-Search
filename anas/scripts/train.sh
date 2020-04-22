#!/usr/bin/env sh
#$-pe gpu 1
#$-l gpu=1
#$-j y
#$-cwd
#$-V
#$-o ./log/train.log
#$-q main.q@yagi11.vision.is.tohoku
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/anhcda/ANAS/anas/train_net.py \
--batch_size 128 \
--note exp1 \
--net_type resnet56 \
>> /home/anhcda/ANAS/anas/outputs/train_output.txt
