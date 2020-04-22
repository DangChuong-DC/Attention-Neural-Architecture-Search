import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from se_resnet_cifar import se_resnet20, se_resnet56, se_resnet110

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model_path', type=str, default='/', help='path of pretrained model')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/ANAS/data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')
parser.add_argument('--resnet_type', type=str, default='20', help='clarify ResNet type: supporting 20, 56, 110')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    num_gpus = torch.cuda.device_count()
    logging.info('Testing with %d GPU(s)', num_gpus)

    model = eval("se_resnet%s(num_classes=CIFAR_CLASSES)" % args.resnet_type)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    try:
        utils.load(model, args.model_path)
    except:
        model = model.module
        utils.load(model, args.model_path)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.cifar100:
        _, test_transform = utils._data_transforms_cifar100(args)
    else:
        _, test_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        test_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=test_transform)
    else:
        test_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('TEST ACCURACY: --- %f% ---', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
