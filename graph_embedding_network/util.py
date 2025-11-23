import numpy as np
import torch
import torch.nn.functional as F
import math
import os
from pathlib import Path

class IOStream():
    def __init__(self, path):
        if not os.path.exists(path):
            Path(path).touch(exist_ok=True)
        self.f = open(path, 'a')


    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def logits_to_angle(logits, K=90):
    """
    logits: Tensor, shape (N, K)
    # return: Tensor of predicted angles in [-180, 180)
    return: Tensor of predicted angles in [-90, 90)
    """
    pred_cls = torch.argmax(logits, dim=1)  # (N,)
    pred_angle = pred_cls*(360/K) - 180
    return pred_angle

def get_trans_mat(x):
    batch_size= x.size(0)
    try:
        theta2 =  x[:,0]/180*torch.pi
    except:
        theta2 =  x/180*torch.pi
    sin, cos = torch.sin(theta2), torch.cos(theta2)
    rotate_mat2 = torch.stack(
        (torch.stack((cos, torch.zeros(batch_size).cuda(), -sin), dim=1),
         torch.stack((torch.zeros(batch_size).cuda(), torch.ones(batch_size).cuda(), torch.zeros(batch_size).cuda()),dim=1),
            torch.stack((sin, torch.zeros(batch_size).cuda(),cos), dim=1))
        ,dim=2)

    return rotate_mat2

def trans_minu(minu,trans,td=False):
    rot1 = get_trans_mat(trans)
    new_pos = torch.bmm(rot1, minu[:, :3])
    new_theta = torch.bmm(rot1, minu[:, 3:6])
    minu = torch.cat((new_pos, new_theta), dim=1)
    return minu


# ---------- 配置 ----------
K = 90  # bins
sigma_bins = 3.0  # 用多少个 bin 的 std 来做软标签
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 预计算 bin 中心角 phi_k (0..2pi)
phis = torch.linspace(-math.pi, math.pi, steps=K+1)[:-1].to(device)  # shape (K,)

# ---------- 工具函数 ----------
def ang_diff(a, b):
    # 返回 a-b 的最小有向差在 (-pi, pi]
    # 支持张量（弧度）
    diff = a - b
    return torch.atan2(torch.sin(diff), torch.cos(diff))

def make_soft_target(theta, K=K, sigma_bins=sigma_bins,max_angle=2*math.pi):
    # theta: (N,) 弧度  in [0,2pi)
    # 返回 soft target (N, K)
    phis = torch.linspace(-max_angle/2, max_angle/2, steps=K + 1)[:-1].to(device)  # shape (K,)
    theta = theta/180*np.pi
    N = theta.shape[0]
    # expand
    t = theta.view(N,1)  # (N,1)
    ph = phis.view(1, K)  # (1,K)
    d = ang_diff(t, ph)  # (N,K)
    # sigma in radians: convert sigma_bins -> radians per bin
    rad_per_bin = max_angle/ K
    # rad_per_bin = math.pi / K
    sigma = sigma_bins * rad_per_bin
    ex = torch.exp(-0.5 * (d / sigma)**2)
    ex = ex / (ex.sum(dim=1, keepdim=True) + 1e-12)
    return ex  # (N,K)


# ---------- 自定义 loss：环形交叉熵（soft targets） ----------
def circular_cross_entropy_loss(logits, soft_targets):
    # logits: (N,K), soft_targets: (N,K) sum->1
    logp = F.log_softmax(logits, dim=1)
    # Negative log-likelihood for soft targets:
    loss = - (soft_targets * logp).sum(dim=1).mean()
    return loss

# ---------- 解码函数：soft-expected angle ----------
def decode_angle_from_probs(probs, phis=phis):
    # probs: (N,K), phis:(K,)
    phis = phis.cuda()
    s = torch.sum(probs * torch.sin(phis), dim=1)
    c = torch.sum(probs * torch.cos(phis), dim=1)
    theta = torch.atan2(s, c)  # in (-pi, pi]
    # map to [0, 2pi)
    theta = (theta+math.pi) % (2*math.pi) - math.pi
    return theta/math.pi*180  # (N,)


