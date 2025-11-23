from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from tqdm import tqdm

from util import make_soft_target,circular_cross_entropy_loss

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import FingerLoader_Train,FingerLoader_pretrain
from model import  GraphNetwork
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
import sklearn.metrics as metrics

def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):

    if args.pretrain_2d:
        # 2D预训练
        train_loader = DataLoader(FingerLoader_pretrain(num_points=args.num_points,img_dir=""), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(FingerLoader_pretrain(num_points=args.num_points,img_dir="",training=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    else:
        # 3D微调
        train_loader = DataLoader(FingerLoader_Train(num_points=args.num_points,img_dir=""), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(FingerLoader_Train(num_points=args.num_points,img_dir="",training=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")



    model = GraphNetwork(args).to(device)


    model = nn.DataParallel(model)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    # 将参数映射到新模型的对应部分

    new_state_dict = model.state_dict()
    for name, param in state_dict.items():
        new_name= name
        if new_name in new_state_dict   :
            try:
                new_state_dict[new_name].copy_(param)
            except Exception as e:
                # 获取预训练权重
                # 复制到目标卷积层的前 6 个输入通道
                print(param.size())
                print(e)


    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = nn.TripletMarginLoss(margin=0.5)

    best_test_acc = 0


    for epoch in range(args.epochs):
        ###################
        # Train
        ###################
        train_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        t = tqdm(train_loader, desc='training', ncols=200)
        gpu_count = torch.cuda.device_count()

        batch_size = args.batch_size
        step = int(batch_size / gpu_count)

        for minu1,disc1,minu2,disc2, label,angle1,angle2 in t:
            minu1,disc1,minu2,disc2,label,angle1,angle2 = (minu1.to(device).to(torch.float),disc1.to(device).to(torch.float),
                                                           minu2.to(device).to(torch.float),disc2.to(device).to(torch.float),
                                                           label.to(device).squeeze(), angle1.to(device).to(torch.float),angle2.to(device).to(torch.float))
            minu1 = minu1.permute(0, 2, 1)
            disc1 = disc1.permute(0, 2, 1)
            minu2 = minu2.permute(0, 2, 1)
            disc2 = disc2.permute(0, 2, 1)
            # batch_size = minu1.size()[0]
            opt.zero_grad()
            logits1,trans1 = model(minu1,disc1,label,neg=True,trans=True)
            logits1, negs = logits1
            logits2,trans2 = model(minu2,disc2,label,neg=True,trans=True)
            logits2, negs2 = logits2
            negss = [negs[i*step:(i+1)*step]+step*i for i in range(gpu_count)]
            negs = torch.cat(negss)
            negss2 = [negs2[i*step:(i+1)*step]+step*i for i in range(gpu_count)]
            negs2 = torch.cat(negss2)
            loss = criterion(logits1, logits2,logits1[negs])/2 + criterion(logits2, logits1,logits2[negs2])/2
            soft_targets1 = make_soft_target(angle1)
            soft_targets2 = make_soft_target(angle2)

            loss2 = (circular_cross_entropy_loss(trans1, soft_targets1)+circular_cross_entropy_loss(trans2, soft_targets2))/2
            loss =loss+loss2
            loss.backward()
            opt.step()
            scheduler.step()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(torch.ones(batch_size).cpu().numpy())
            train_pred.append((F.cosine_similarity(logits1,logits2)>F.cosine_similarity(logits1,logits1[negs])).detach().cpu().numpy())
            t.set_description(f'Epoch [{epoch}]')
            t.set_postfix(loss = train_loss/count,acc = np.concatenate(train_pred).sum()*1.0/count)

        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, lr: %.6f' % (epoch,
                                                                 train_loss*1.0/count,
                                                                 train_pred.sum()/len(train_pred),get_lr(opt))
        io.cprint(outstr)

        ###################
        # Test
        ###################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        t = tqdm(test_loader, desc='testing', ncols=200)
        batch_size = args.test_batch_size
        step = int(batch_size / gpu_count)

        for minu1,disc1,minu2,disc2, label,angle1,angle2 in t:
            minu1,disc1,minu2,disc2, label = minu1.to(device).to(torch.float),disc1.to(device).to(torch.float),minu2.to(device).to(torch.float),disc2.to(device).to(torch.float), label.to(device).squeeze()
            minu1 = minu1.permute(0, 2, 1)
            disc1 = disc1.permute(0, 2, 1)
            minu2 = minu2.permute(0, 2, 1)
            disc2 = disc2.permute(0, 2, 1)

            batch_size = minu1.size()[0]
            logits1,negs = model(minu1,disc1,label,neg=True)
            logits2 = model(minu2,disc2,label)
            negss = [negs[i*step:(i+1)*step]+step*i for i in range(gpu_count)]
            negs = torch.cat(negss)
            loss = criterion(logits1, logits2,logits1[negs])
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(torch.ones(batch_size).cpu().numpy())

            test_pred.append((F.cosine_similarity(logits1,logits2)>F.cosine_similarity(logits1,logits1[negs])).detach().cpu().numpy())
            t.set_description(f'Epoch [{epoch}]')
            t.set_postfix(loss = test_loss/count,acc = np.concatenate(test_pred).sum()*1.0/count)

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch,
                                                                     test_loss*1.0/count,
                                                                     test_pred.sum()/len(test_pred))
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t145_dmd' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=200,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=15, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--k2', type=int, default=9, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--pretrain_2d', type=bool, default=False,
                        help='if use 2D fingerprints to pretrain or use 3D fingerprints to finetune')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp/models/model_pretrain.pt', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
