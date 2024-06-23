import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
from typing import Tuple
import tqdm
import numpy as np

import torchvision
from torchvision import models
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append('../../..')
from dalib.adaptation.mcd import entropy
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import tsne, a_distance
from examples.domain_adaptation.image_classification import na_utils
import examples.domain_adaptation.image_classification.na_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def classifier_discrepancy(predictions1, predictions2):
    return torch.mean(torch.abs(predictions1 - predictions2))
class ImageClassifierHead(nn.Module):

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim=1000, pool_layer=None):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.head(inputs)
        return output

class Feature_extrator(nn.Module):
    def __init__(self, model_name, pre):
        super(Feature_extrator, self).__init__()
        self.G = utils.get_model(model_name, pretrain=pre)
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )

        self.out_features = self.G.out_features

    def forward(self, inputs):
        x = self.G(inputs)
        outputs = self.pool_layer(x)
        return outputs

def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()
def CDD(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    mul = predictions1.transpose(0, 1).mm(predictions2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
def data_load(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    aug_transform = na_utils.TransformFixMatch(crop_size=224)
    if args.source == ['A']:
        s_dset_path = "/home/b230/data/DA/zxy/Art.txt"
    elif args.source == ['C']:
        s_dset_path = "/home/b230/data/DA/zxy/Clipart.txt"
    elif args.source == ['P']:
        s_dset_path = "/home/b230/data/DA/zxy/Product.txt"
    else:
        s_dset_path = "/home/b230/data/DA/zxy/Real_World.txt"

    if args.target == ['A']:
        t_dset_path = "/home/b230/data/DA/zxy/Art.txt"
    elif args.target == ['C']:
        t_dset_path = "/home/b230/data/DA/zxy/Clipart.txt"
    elif args.target == ['P']:
        t_dset_path = "/home/b230/data/DA/zxy/Product.txt"
    else:
        t_dset_path = "/home/b230/data/DA/zxy/Real_World.txt"
    print(s_dset_path)
    print(t_dset_path)
    source_set = na_utils.ObjectImage('', s_dset_path, train_transform)
    target_set = na_utils.ObjectImage_mul('', t_dset_path, train_transform)
    test_set = na_utils. ObjectImage('', t_dset_path, test_transform)
    target_aug_set = na_utils.ObjectImage_mul('', t_dset_path, aug_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    dset_loaders["target_aug"] = torch.utils.data.DataLoader(target_aug_set, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=args.workers, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.workers, drop_last=False)

    return dset_loaders

def collect_feature(data_loader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    dset_loaders = data_load(args)
    train_source_iter = iter(dset_loaders["source"])
    # train_target_iter = iter(dset_loaders["target"])
    train_target_iter = iter(dset_loaders["target_aug"])
    val_loader = test_loader = dset_loaders["test"]
    # create model
    print("=> using model '{}'".format(args.arch))
    G = Feature_extrator(args.arch, pre=not args.scratch).to(device)  # feature extractor
    # two image classifier heads
    pool_layer = nn.Identity() if args.no_pool else None
    F1 = ImageClassifierHead(G.out_features, 65, args.bottleneck_dim, pool_layer).to(device)
    F2 = ImageClassifierHead(G.out_features, 65, args.bottleneck_dim, pool_layer).to(device)
    # define optimizer
    # the learning rate is fixed according to origin paper
    optimizer_g = SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = SGD([
        {"params": F1.parameters()},
        {"params": F2.parameters()},
    ], momentum=0.9, lr=args.lr, weight_decay=0.0005)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        G.load_state_dict(checkpoint['G'])
        F1.load_state_dict(checkpoint['F1'])
        F2.load_state_dict(checkpoint['F2'])

    # analysis the model
    if args.phase == 'analysis':
        print("analysis")
        feature_extractor = nn.Sequential(G).to(device)
        source_feature = collect_feature(train_source_iter, feature_extractor, device)
        target_feature = collect_feature(val_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return
    if args.phase == 'test':
        acc1 = validate(test_loader, G, F1, F2, args)
        print(acc1)
        return
    if args.phase == 'continue':
        print("continue train")


    # start training
    if args.phase == 'train':
        best_acc1 = 0.
        best_results = None
    #memory bank
    mem_fea = torch.rand(len(dset_loaders["target"].dataset), 2048).to(device)
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls_1 = torch.ones(len(dset_loaders["target"].dataset), 65).cuda() / 65
    mem_cls_2 = torch.ones(len(dset_loaders["target"].dataset), 65).cuda() / 65
    # mem_cls_1 = torch.rand(len(dset_loaders["target"].dataset), 65).to(device)
    # mem_cls_2 = torch.rand(len(dset_loaders["target"].dataset), 65).to(device)

    for epoch in range(args.epochs):
        # train for one epoch
        mem_fea, mem_cls_1, mem_cls_2 = train(dset_loaders, mem_fea, mem_cls_1, mem_cls_2, train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, args)

        # remember best acc@1 and save checkpoint
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict()
        }, logger.get_checkpoint_path('latest'))
        if max(results) > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(results)
            best_results = results

    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))

    # evaluate on test set
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    F2.load_state_dict(checkpoint['F2'])
    results = validate(test_loader, G, F1, F2, args)
    print("test_acc1 = {:3.1f}".format(max(results)))

    logger.close()


def train(dset_loaders, mem_fea, mem_cls_1, mem_cls_2, train_source_iter, train_target_iter,
          G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace):

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    # losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    nums_1 = AverageMeter('num1', ':3.2f')
    nums_2 = AverageMeter('num2', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, trans_losses, nums_1, nums_2, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))
    class_weight_src = torch.ones(65, ).to(device)

    # switch to train mode
    G.train()
    F1.train()
    F2.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        try:
            x_s, labels_s = next(train_source_iter)
        except:
            train_source_iter = iter(dset_loaders["source"])
            x_s, labels_s = next(train_source_iter)
        try:
            x_t, labels_t, idx = next(train_target_iter)
        except:
            train_target_iter = iter(dset_loaders["target_aug"])
            x_t, labels_t, idx = next(train_target_iter)
        x_s = x_s.to(device)
        x_t_w = x_t[0]
        x_t_s = x_t[1]
        x_t_w = x_t_w.to(device)
        x_t_s = x_t_s.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x_1 = torch.cat((x_s, x_t_w), dim=0)
        x = torch.cat((x_1, x_t_s), dim=0)
        assert x.requires_grad is False

        # measure data loading time
        data_time.update(time.time() - end)
        eff = (i + args.iters_per_epoch * epoch) / (args.iters_per_epoch * args.epochs)
        # Step A train all networks to minimize loss on source domain
        G.train()
        F1.train()
        F2.train()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x)
        f_s, f_t, f_t_s = g.chunk(3, dim=0)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t, y1_t_s = y_1.chunk(3, dim=0)
        y2_s, y2_t, y2_t_s = y_2.chunk(3, dim=0)
        preds_1 = torch.softmax(y1_t.detach(), dim=-1)
        preds_2 = torch.softmax(y2_t.detach(), dim=-1)
        max_probs_1, pseudo_label_1 = torch.max(preds_1, dim=-1)
        max_probs_2, pseudo_label_2 = torch.max(preds_2, dim=-1)
        num_1 = torch.tensor(pseudo_label_1[max_probs_1.ge(args.fixmatch_threshold)].size(0)).float()
        num_2 = torch.tensor(pseudo_label_2[max_probs_2.ge(args.fixmatch_threshold)].size(0)).float()
        Lu_cross = ((
                 F.cross_entropy(y1_t_s, pseudo_label_2, reduction="none") * max_probs_2.ge(
                 args.fixmatch_threshold).float()
             ).sum() + (F.cross_entropy(y2_t_s, pseudo_label_1, reduction="none") * max_probs_1.ge(
             args.fixmatch_threshold).float()).sum()) / args.batch_size

        Lu_self = ((
                      F.cross_entropy(y1_t_s, pseudo_label_1, reduction="none") * max_probs_1.ge(
                  args.fixmatch_threshold).float()
              ).sum() + (F.cross_entropy(y2_t_s, pseudo_label_2, reduction="none") * max_probs_2.ge(
            args.fixmatch_threshold).float()).sum()) / args.batch_size
        Lu = (Lu_cross + Lu_self) / 2.
        outputs_target_1, outputs_target_2 = y1_t, y2_t
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        src_1 = CrossEntropyLabelSmooth(num_classes=65, epsilon=args.smooth)(y1_s, labels_s)
        weight_src_1 = class_weight_src[labels_s].unsqueeze(0)
        loss = torch.sum(weight_src_1 * src_1) / (torch.sum(weight_src_1).item())

        src_2 = CrossEntropyLabelSmooth(num_classes=65, epsilon=args.smooth)(y2_s, labels_s)
        weight_src_2 = class_weight_src[labels_s].unsqueeze(0)
        loss = loss + torch.sum(weight_src_2 * src_2) / (torch.sum(weight_src_2).item())

        loss = loss + (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy
        # f_s, f_t = g.chunk(2, dim=0)
        features_target = f_t
        #NA
        dis = -torch.mm(features_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.max(dis)
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1 / args.K

        weight_1, pred_1 = torch.max(w.mm(mem_cls_1), 1)
        weight_2, pred_2 = torch.max(w.mm(mem_cls_2), 1)
        loss_1 = nn.CrossEntropyLoss(reduction='none')(outputs_target_1, pred_1)
        classifier_loss = torch.sum(weight_1 * loss_1) / (torch.sum(weight_1).item())
        loss_2 = nn.CrossEntropyLoss(reduction='none')(outputs_target_2, pred_2)
        classifier_loss = classifier_loss + torch.sum(weight_2 * loss_2) / (torch.sum(weight_2).item())
        loss = loss + eff * classifier_loss * args.self_weight + Lu

        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        #memory bank update
        G.eval()
        F1.eval()
        F2.eval()
        with torch.no_grad():
            # g = G(x_t)
            g = G(x_t_w)
            features_target = g
            outputs_target_1 = F1(g)
            outputs_target_2 = F2(g)
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out_1 = nn.Softmax(dim=1)(outputs_target_1)
            softmax_out_2 = nn.Softmax(dim=1)(outputs_target_2)
            outputs_target_1 = softmax_out_1 ** 2 / ((softmax_out_1 ** 2).sum(dim=0))  # 通过温度来直接锐化预测输出
            outputs_target_2 = softmax_out_2 ** 2 / ((softmax_out_2 ** 2).sum(dim=0))
        mem_fea[idx] = features_target.clone()
        mem_cls_1[idx] = outputs_target_1.clone()
        mem_cls_2[idx] = outputs_target_2.clone()

        G.train()
        F1.train()
        F2.train()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        g = G(x_1)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        outputs_target_1, outputs_target_2 = y1_t, y2_t
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        src_1 = CrossEntropyLabelSmooth(num_classes=31, epsilon=args.smooth)(y1_s, labels_s)
        weight_src_1 = class_weight_src[labels_s].unsqueeze(0)
        loss = torch.sum(weight_src_1 * src_1) / (torch.sum(weight_src_1).item())
        src_2 = CrossEntropyLabelSmooth(num_classes=31, epsilon=args.smooth)(y2_s, labels_s)
        weight_src_2 = class_weight_src[labels_s].unsqueeze(0)
        loss = loss + torch.sum(weight_src_2 * src_2) / (torch.sum(weight_src_2).item())
        loss = loss + (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy
        loss = loss - CDD(y1_t, y2_t) * args.trade_off

        f_s, f_t = g.chunk(2, dim=0)
        features_target = f_t
        # # NA
        dis = -torch.mm(features_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.max(dis)
        _, p1 = torch.sort(dis, dim=1)
        #
        w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1 / args.K
        #
        weight_1, pred_1 = torch.max(w.mm(mem_cls_1), 1)
        weight_2, pred_2 = torch.max(w.mm(mem_cls_2), 1)
        loss_1 = nn.CrossEntropyLoss(reduction='none')(outputs_target_1, pred_1)
        classifier_loss = torch.sum(weight_1 * loss_1) / (torch.sum(weight_1).item())
        loss_2 = nn.CrossEntropyLoss(reduction='none')(outputs_target_2, pred_2)
        classifier_loss = classifier_loss + torch.sum(weight_2 * loss_2) / (torch.sum(weight_2).item())
        loss = loss + eff * classifier_loss * args.self_weight

        loss.backward()
        optimizer_f.step()
        # memory bank update
        G.eval()
        F1.eval()
        F2.eval()
        with torch.no_grad():
            g = G(x_t_w)
            features_target = g
            outputs_target_1 = F1(g)
            outputs_target_2 = F2(g)
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out_1 = nn.Softmax(dim=1)(outputs_target_1)
            softmax_out_2 = nn.Softmax(dim=1)(outputs_target_2)
            outputs_target_1 = softmax_out_1 ** 2 / ((softmax_out_1 ** 2).sum(dim=0))
            outputs_target_2 = softmax_out_2 ** 2 / ((softmax_out_2 ** 2).sum(dim=0))
        mem_fea[idx] = features_target.clone()
        mem_cls_1[idx] = outputs_target_1.clone()
        mem_cls_2[idx] = outputs_target_2.clone()

        # Step C train genrator to minimize discrepancy
        for k in range(args.num_k):
            G.train()
            F1.train()
            F2.train()
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            # g = G(x)
            g = G(x_1)

            y_1 = F1(g)
            y_2 = F2(g)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            y1_t_s, y2_t_s = F.softmax(y1_t_s, dim=1), F.softmax(y2_t_s, dim=1)
            CDD_loss = (CDD(y1_t, y2_t)) * args.trade_off
            D_loss = CDD_loss + (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy
            D_loss.backward()
            optimizer_g.step()

        cls_acc = accuracy(y1_s, labels_s)[0]
        tgt_acc = accuracy(y1_t, labels_t)[0]

        # losses.update(loss.item(), x_s.size(0))
        nums_1.update(num_1, 1)
        nums_2.update(num_2, 1)
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t_w.size(0))
        trans_losses.update(CDD_loss.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            print("eff:", eff)
    return mem_fea, mem_cls_1, mem_cls_2


def validate(val_loader: DataLoader, G: nn.Module, F1:ImageClassifierHead,
             F2: ImageClassifierHead, args: argparse.Namespace) -> Tuple[float, float]:
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_1, top1_2],
        prefix='Test: ')

    # switch to evaluate mode
    G.eval()
    F1.eval()
    F2.eval()

    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            g = G(images)
            y1, y2 = F1(g), F2(g)

            # measure accuracy and record loss
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            if confmat:
                confmat.update(target, y1.argmax(1))
            top1_1.update(acc1.item(), images.size(0))
            top1_2.update(acc2.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))
        if confmat:
            print(confmat.format(args.class_names))

    return top1_1.avg, top1_2.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCNA for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office-home', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office-home)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=0.01, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off-entropy', default=0.01, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('--num-k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='bcmd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'continue'
                                                                                ],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--temperature', default=2.5, type=float, help='parameter temperature scaling')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--self_weight', type=float, default=1.)
    parser.add_argument('--fixmatch_threshold', default=0.95, type=float)
    parser.add_argument('--mul_weight', type=float, default=1.)
    args = parser.parse_args()
    main(args)
