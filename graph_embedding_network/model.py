import torch
import torch.nn as nn
import torch.nn.functional as F
from util import trans_minu,logits_to_angle


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=10, idx=None,para=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if para is not None:
        feature = torch.cat(((feature-x)*para[0], x*para[1]), dim=3).permute(0, 3, 1, 2).contiguous()

    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature


class GraphEmbeddingNetwork(nn.Module):
    def __init__(self, args, output_channels=256,in_channel=128,minu=True):


        super(GraphEmbeddingNetwork, self).__init__()
        self.args = args
        self.minu = minu
        if minu:
            self.k = args.k
        else:
            self.k = args.k2

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)

        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(256)

        self.bn5 = nn.BatchNorm1d(args.emb_dims)



        self.conv1 = nn.Sequential(nn.Conv2d(in_channel*2, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(640, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)

        self.bn7 = nn.BatchNorm1d(256)

        self.linear2_1 = nn.Linear(256, 256, bias=False)

        self.linear2_2 = nn.Linear(256, 256, bias=False)

        self.bn8 = nn.BatchNorm1d(256)



        self.linear3 = nn.Linear(256, output_channels, bias=False)


        self.res1 = nn.Sequential(nn.Conv2d(in_channel*2, 128, (1, 1), 1, bias=False))
        self.res2 = nn.Sequential(nn.Conv2d(128*2, 128, (1, 1), 1, bias=False),nn.BatchNorm2d(64))

        self.res3 = nn.Sequential(nn.Conv2d(128*2, 128, (1, 1), 1, bias=False))

        self.res4 = nn.Sequential(nn.Conv2d(256, 256, (1, 1), 1, bias=False),nn.BatchNorm2d(256))

        self.pos_ecode1 = nn.Conv1d(6, 128,kernel_size=1)
        self.pos_ecode2 =  nn.Conv1d(768, 128,kernel_size=1)
        self.para = nn.Parameter(torch.randn(8, 2))
        # 初始化目标卷积层的权重
    def forward(self, disc,x,label,neg=False,angle=False):
        batch_size = disc.size(0)
        x = x +self.pos_ecode1(disc)
        x = get_graph_feature(x, k=self.k,para=[self.para[0,0],self.para[0,1]])
        x = self.conv1(x) + self.res1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]


        x = get_graph_feature(x1, k=self.k,para=[self.para[2,0],self.para[2,1]])
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k,para=[self.para[4,0],self.para[4,1]])
        x = self.conv3(x) + self.res3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k,para=[self.para[6,0],self.para[6,1]])
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        if self.minu:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)

            x = F.leaky_relu_(self.bn6(self.linear1(x)),negative_slope=0.2)
            x = F.leaky_relu_(self.bn7(self.linear2(x)),negative_slope=0.2)
        if neg:
            if not self.minu:
                x = x[:,:,0]
            if angle:
                dot = torch.mm(x,x.T)
                len = (x**2).sum(dim=1).unsqueeze(1)
                Msim = dot / torch.sqrt(torch.mm(len, len.T))
                for i in range(label.size(0)):
                    Msim[i][label==label[i]] = -1
                negs = torch.argmax(Msim,dim=0)
            else:
                dot = torch.mm(x, x.T)
                len = (x ** 2).sum(dim=1).unsqueeze(1)
                Msim = len - 2 * dot + len.T
                for i in range(label.size(0)):
                    Msim[i][label == label[i]] = torch.inf
                negs = torch.argmin(Msim, dim=0)
                return x, negs
            return x,negs
        else:
            if not self.minu:
                x = x[:,:,0]
            return x


class PoseCorrectionNetwork(nn.Module):
    def __init__(self, args, output_channels=90,in_channel=6,minu=True):
        super(PoseCorrectionNetwork, self).__init__()
        self.args = args
        self.minu = minu
        if minu:
            self.k = args.k
        else:
            self.k = args.k2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)


        self.conv1 = nn.Sequential(nn.Conv2d(in_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128, bias=False)

        self.bn7 = nn.BatchNorm1d(128)

        self.linear2_1 = nn.Linear(128, 64, bias=False)

        self.linear2_2 = nn.Linear(64, 32, bias=False)

        self.bn8 = nn.BatchNorm1d(32)



        self.linear3 = nn.Linear(32, output_channels, bias=False)
        # self.pos_ecode1 = nn.Conv1d(6, 128,kernel_size=1)


    def forward(self, x,disc):
        batch_size = x.size(0)
        # x = torch.concatenate((x[:,:2],torch.zeros_like(x[:,2]).unsqueeze(1),x[:,2].unsqueeze(1),torch.zeros_like(x[:,:2])),dim=1)
        # x = torch.concatenate((self.pos_ecode1(x[:,:3]),self.pos_ecode2(x[:,3:6])),dim=1)
        # x = x[:,3:6] + torch.bmm(self.pos_ecode1.unsqueeze(0).repeat(256,1,1),x[:,3:])
        # x = torch.cat([x,disc],dim=1)
        # x = x+ self.pos_ecode1(disc)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        if self.minu:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)

            x = F.leaky_relu_(self.bn6(self.linear1(x)),negative_slope=0.2)
            x = F.leaky_relu_(self.bn7(self.linear2(x)),negative_slope=0.2)
            x = F.gelu(self.bn8(self.linear2_2(self.linear2_1(x))))
            x = self.linear3(x)
        return x


class GraphNetwork(torch.nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()
        self.backbone1 = PoseCorrectionNetwork(args)
        self.backbone2 = GraphEmbeddingNetwork(args)
        self.minu =True

    def forward(self,data,disc,label,neg=False,trans=False):
        trans1 = self.backbone1(data,disc)
        trans2 = logits_to_angle(trans1)
        data = trans_minu(data, trans2)
        x =self.backbone2(data, disc, label, neg)
        if trans:
            return x,trans1
        else:
            return x



