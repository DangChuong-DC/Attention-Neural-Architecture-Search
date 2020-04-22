#!/usr/bin/env sh
#$-pe gpu 1
#$-l gpu=1
#$-j y
#$-cwd
#$-V
#$-o ./log/train.log
#$-q main.q@yagi04.vision.is.tohoku
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python /home/anhcda/ANAS/se_resnet_cifar/train.py \
--resnet_type 20 \
--batch_size 128 \
--epochs 180 \
--note exp1 \
>> /home/anhcda/ANAS/se_resnet_cifar/outputs/train_output.txt
