from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
import torch
import torch.nn as nn
from data import FingerLoader_Identify
from model import GraphNetwork
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
import scipy.io
from tqdm import tqdm
torch.manual_seed(3047)

from model import logits_to_angle

def compute_angle_diff_cfpose(angle_gt,angle_pred1):
    angle_gt = angle_gt[:120,:,1]
    angle_pred1 = angle_pred1.reshape((120,10))
    diff = np.abs(angle_pred1 - angle_gt).mean(axis=0)
    print(diff)
    print(np.mean(diff))

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
def test(args):
    angles_gt = scipy.io.loadmat("/mnt/sdb/jyw_data/dgcnn/angle_refine_cfpose_gt.mat")['angle_refine']

    test_loader = DataLoader(FingerLoader_Identify(num_points=args.num_points,
                                                                img_dir="/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_df_3d",
                                                                img_dir2="/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_feature",uwa=False), num_workers=8,
                             batch_size=args.test_batch_size, drop_last=False,shuffle=False)
    device = torch.device("cuda" if args.cuda else "cpu")


    model = GraphNetwork(args).to(device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    count = 0.0
    score_pred = np.zeros((1200, 256))
    angles_pred = np.zeros((1200))
    angles_pred2 = np.zeros((1200))


    labels = []
    labels2 = []
    for i in range(1, 121):
        for j in range(1, 11):
            labels.append(i)
            labels2.append(i)
    labels = np.array(labels)
    labels2 = np.array(labels2)
    t = tqdm(test_loader, desc='training', ncols=200)

    for  minu1,disc1, label,name in t:
        name = [i.split("/")[-1] for i in name]
        idxs = [(int(i.split(".")[0].split("_")[0]) - 1) * 10 + (int(i.split(".")[0].split("_")[1]) - 1)
                for i in name]
        minu1, disc1, label = minu1.to(device).to(torch.float), disc1.to(device).to(torch.float), label.to(
            device).squeeze()
        minu1 = minu1.permute(0, 2, 1)
        disc1 = disc1.permute(0, 2, 1)
        batch_size = minu1.size()[0]
        logits1,trans1 = model(minu1, disc1, label,trans=True)
        angles = logits_to_angle(trans1)
        count += batch_size

        score_pred[idxs] = ((logits1).detach().cpu().numpy())
        angles_pred[idxs] = angles.detach().cpu().numpy()

    compute_angle_diff_cfpose(angles_gt,angles_pred)
    score_pred = torch.tensor(score_pred)
    dot = torch.mm(score_pred, score_pred.T)
    len1 = (score_pred ** 2).sum(dim=1).unsqueeze(1)
    diff_angle = np.abs(angles_pred[:, None] - angles_pred2[None, :])
    Msim = np.abs(dot / torch.sqrt(torch.mm(len1, len1.T)))
    print(Msim)
    shape = Msim.shape
    print(shape)
    Msim= torch.nan_to_num(Msim)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i==j:
                Msim[i,j]=-np.inf



    print(Msim)
    negs = torch.argmax(Msim, dim=1)
    negs2 = torch.topk(Msim, dim=1,k=20,largest=True)[1]
    print(negs2)
    print(torch.topk(Msim, dim=1,k=20,largest=True)[0])
    print(torch.tensor(labels2)[negs])
    print(torch.tensor(labels))
    acc = (torch.tensor(labels) == torch.tensor(labels2)[negs]).sum() / len(labels)
    print(acc)

    out2 = torch.tensor(labels2)[negs2]

    rank_n=[]
    for i in range(20):
        acc2 = torch.tensor([label in out2[idx][:i+1] for idx,label in enumerate(torch.tensor(labels))]).sum()  / len(labels)
        print(acc2)
        rank_n.append(acc2)


    count_out= []
    for idx, label in enumerate(torch.tensor(labels)):
        count = sum(1 for num in out2[idx][:20] if num == label)
        count_out.append(count)
    print(np.mean(count_out))
    score_pred_true=[]
    angle_pred_true=[]
    score_pred_false=[]
    angle_pred_false=[]

    for i in range(len(labels)-1):
        for j in range(i+1,len(labels2)):
            if Msim[i][j]!=-np.inf:
                if labels[i]==labels2[j]:
                    score_pred_true.append(Msim[i][j])
                    angle_pred_true.append(diff_angle[i][j])
                else:
                    score_pred_false.append(Msim[i][j])
                    angle_pred_false.append(diff_angle[i][j])

    print(len(score_pred_true))
    print(len(score_pred_false))


    scipy.io.savemat("test_cfpose_det.mat", mdict={'cscores1': score_pred_true,'cscores2': score_pred_false})
    scipy.io.savemat("test_cfpose_angle.mat", mdict={'angles1': angle_pred_true,'angles2': angle_pred_false})

    scipy.io.savemat("test_cfpose_cmc.mat", mdict={'rank1':rank_n})
    scipy.io.savemat("test_cfpose_msim.mat", mdict={'msim':np.array(Msim)})



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
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

    _init_()



    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    test(args)
