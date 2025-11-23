from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
import torch
import torch.nn as nn
from data import FingerLoader_Identify_ZJU

from model import GraphNetwork
import numpy as np
from torch.utils.data import DataLoader
import scipy.io
from tqdm import tqdm
from model import logits_to_angle


def test(args):
    train_loader = DataLoader(FingerLoader_Identify_ZJU(num_points=args.num_points, img_dir=""), num_workers=8,
                              batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(FingerLoader_Identify_ZJU(num_points=args.num_points, img_dir="", data1=False), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")


    model = GraphNetwork(args).to(device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    t = tqdm(train_loader, desc='testing', ncols=200)
    true_score = []
    true_angles= []
    false_score = []
    false_angles = []
    count = 0.0
    for minu1, disc1, minu2, disc2, label in t:
        minu1, disc1, minu2, disc2, label = minu1.to(device).to(torch.float), disc1.to(device).to(
            torch.float), minu2.to(device).to(torch.float), disc2.to(device).to(torch.float), label.to(device).squeeze()
        minu1 = minu1.permute(0, 2, 1)
        disc1 = disc1.permute(0, 2, 1)
        minu2 = minu2.permute(0, 2, 1)
        disc2 = disc2.permute(0, 2, 1)

        batch_size = minu1.size()[0]
        logits1,trans1 = model(minu1, disc1, label,trans=True)
        angles1 = logits_to_angle(trans1)
        logits2,trans2 = model(minu2, disc2, label,trans=True)
        angles2 = logits_to_angle(trans2)
        scores = (logits1*logits2).sum(dim=1) /torch.sqrt((logits1**2).sum(dim=1)*(logits2**2).sum(dim=1))
        angles1 = angles1.detach().cpu().numpy()
        angles2 = angles2.detach().cpu().numpy()
        diff_angle = np.abs(angles1[:] - angles2[:])

        scores = list(scores.cpu().detach().numpy())
        diff_angle = list(diff_angle)
        true_score.extend(scores)
        true_angles.extend(diff_angle)
        count += batch_size
    t = tqdm(test_loader, desc='testing', ncols=200)

    for minu1, disc1, minu2, disc2, label in t:
        minu1, disc1, minu2, disc2, label = minu1.to(device).to(torch.float), disc1.to(device).to(
            torch.float), minu2.to(device).to(torch.float), disc2.to(device).to(torch.float), label.to(device).squeeze()
        minu1 = minu1.permute(0, 2, 1)
        disc1 = disc1.permute(0, 2, 1)
        minu2 = minu2.permute(0, 2, 1)
        disc2 = disc2.permute(0, 2, 1)

        batch_size = minu1.size()[0]
        logits1,trans1 = model(minu1, disc1, label,trans=True)
        angles1 = logits_to_angle(trans1)
        logits2,trans2 = model(minu2, disc2, label,trans=True)
        angles2 = logits_to_angle(trans2)
        scores = (logits1*logits2).sum(dim=1) /torch.sqrt((logits1**2).sum(dim=1)*(logits2**2).sum(dim=1))
        angles1 = angles1.detach().cpu().numpy()
        angles2 = angles2.detach().cpu().numpy()
        diff_angle = np.abs(angles1[:] - angles2[ :])

        scores = list(scores.cpu().detach().numpy())
        diff_angle = list(diff_angle)

        false_score.extend(scores)
        false_angles.extend(diff_angle)
        count += batch_size
    print(len(true_score))
    print(len(false_score))
    true_score = np.array(true_score)
    true_angles = np.array(true_angles)
    false_score = np.array(false_score)
    false_angles = np.array(false_angles)


    scipy.io.savemat("/mnt/sdb/jyw_data/dgcnn/test_zju_det.mat", mdict={'cscores1': np.array(true_score),'cscores2': np.array(false_score)})
    scipy.io.savemat("/mnt/sdb/jyw_data/dgcnn/test_uwa_angle.mat", mdict={'angles1': true_angles,'angles2': false_angles})


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=200,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--k2', type=int, default=9, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='checkpoints/exp/models/model.pt', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()



    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    test(args)
