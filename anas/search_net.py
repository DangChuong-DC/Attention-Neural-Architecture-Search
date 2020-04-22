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
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from res_att_search_cifar import att_resnet_cifar
from architect import Architect_m, Architect_s
from genotypes import ATT_PRIMITIVES


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/home/anhcda/ANAS/anas/checkPoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/ANAS/data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', type=float, default=0., help='dropout rate of skip connect')   ### default: 0.3
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--unrolled', action='store_true', default=False, help='use unrolled')
parser.add_argument('--net_type', type=str, default='resnet20', help='define resnet type, default: resnet20 for search')

args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

### MAIN--start--
def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    seed = args.seed
    logging.info('Using the random seed of %d for searching...' % seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logging.info("args = %s", args)
    num_gpus = torch.cuda.device_count()
    logging.info('Training with %d GPU(s)', num_gpus)

    # build Network
    # default as ResNet20 since the constrain of GPU memory when doing search process
    resnet_types = {'resnet20': 3, 'resnet32': 5, 'resnet44': 7, 'resnet56': 9, 'resnet110': 18}
    n_sizes = resnet_types[args.net_type]

    logging.info('Number of attentional residual block(s): %s', n_sizes * 3)
    model = att_resnet_cifar(n_size=n_sizes, no_gpus=num_gpus, num_classes=CIFAR_CLASSES)

    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    #  prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if num_gpus > 1:
        optimizer = torch.optim.SGD(
                model.module.net_parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        architect = Architect_m(model, args)
    else:
        optimizer = torch.optim.SGD(
                model.net_parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        architect = Architect_s(model, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    epochs = args.epochs
    scale_factor = 0.19
    BEST_accVal = 0.0
    for epoch in range(epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()

        # training
        if args.dropout_rate > 0.:
            drop_rate = args.dropout_rate * np.exp(-epoch * scale_factor)
            if num_gpus > 1:
                model.module.update_p(drop_rate)
            else:
                model.update_p(drop_rate)

        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, num_gpus)

        logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds', epoch_duration)
        # validation
        if epochs - epoch < 10:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)
            if valid_acc > BEST_accVal:
                BEST_accVal = valid_acc

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('BEST VALID ACCURACY IS: %f', BEST_accVal)

    if num_gpus > 1:
        genotype = model.module.genotype()
    else:
        genotype = model.genotype()
    logging.info('______________________________________________\nFinal genotype = %s', genotype)
    with open('{}/result.txt'.format(args.save), 'w') as file:
        file.write(str(genotype))

    logging.info('____________________END_______________________')


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, gpus):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
        # the training when using PyTorch 0.4 and above.
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        if gpus > 1:
            nn.utils.clip_grad_norm_(model.module.net_parameters(), args.grad_clip)
        else:
            nn.utils.clip_grad_norm_(model.net_parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
