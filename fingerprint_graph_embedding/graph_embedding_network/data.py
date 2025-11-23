import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io

def get_3d_rotate_mat():
    # if np.random.random()>0.5:
    #     theta1 = np.pi/4
    # else:
    #     theta1 = -np.pi/4
    theta1 = np.random.uniform(low=-1,high=1)*np.pi/12
    sin, cos = np.sin(theta1), np.cos(theta1)
    rotate_mat1 = np.array([[cos, -sin,0], [sin, cos,0],[0,0,1]])
    # if np.random.random()>0.5:
    #     theta2 = np.pi/4
    # else:
    #     theta2 = -np.pi/4
    # else:
    #     theta2 = 0
    theta2 = np.random.uniform(low=-1,high=1)*np.pi/6
    sin, cos = np.sin(theta2), np.cos(theta2)
    rotate_mat2 = np.array([[cos, 0,-sin], [0, 1,0],[sin,0,cos]])

    # sin, cos = np.sin(theta2), np.cos(theta2)
    # rotate_mat2 = np.array([[cos, 0,sin], [0, 1,0],[-sin,0,cos]])
    # if np.random.random()>0.5:
    #     theta3 = np.pi/4
    # else:
    #     theta3 = -np.pi/4
    theta3 = np.random.uniform(low=-1,high=1)*np.pi/12
    sin, cos = np.sin(theta3), np.cos(theta3)
    rotate_mat3 = np.array([[1, 0,0], [0, cos,-sin],[0,sin,cos]])
    rote_t = np.dot(rotate_mat2,rotate_mat1)
    rote_t2 = np.dot(rotate_mat3,rote_t)
    return rote_t2


def from_2d_to_3d(minu):
    cx = minu[:,0].mean()
    cy = minu[:,1].mean()
    minu = minu - np.array([cx,cy,0])

    max_len = np.max(np.sum(minu[:,:2]**2,axis=1))
    z = np.sqrt(max_len+70000-(np.sum(minu[:,:2]**2,axis=1)))
    fx = minu[:,1]/z
    fy = minu[:,0]/z

    dz = np.cos(minu[:,2] / 180 * np.pi)
    # theta = np.arctan(fy/fx)
    dx = np.sin(minu[:,2] / 180 * np.pi) * fx / np.sqrt(fx**2+fy**2)
    dy = np.sin(minu[:,2] / 180 * np.pi) * fy / np.sqrt(fx**2+fy**2)
    len = 25 / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    rotate = np.array((dz * len, dx * len, dy * len)).T
    minu_out = np.concatenate((minu[:,:2],
                               np.expand_dims(z,axis=1),
                               rotate
                               ),axis=1)
    return minu_out

class FingerLoader_Train(Dataset):
    def __init__(self, num_points, img_dir,training=True):
        self.BASE_DIR = img_dir
        self.training = training
        self.angle_refine = scipy.io.loadmat("/mnt/sdc/jyw_data/angle_refine_cfpose.mat")['angle_refine']
        self.angle_refine_uwa = scipy.io.loadmat("/mnt/sdc/jyw_data/angle_refine_uwa.mat")['angle_refine']
        self.pairs = self._read_data()
        self.num_points = num_points

    def __getitem__(self, item):
        name1 = self.pairs[item][0].split("/")[-1].split('_')
        name2 = self.pairs[item][2].split("/")[-1].split('_')
        angle1 = 0
        angle2 = 0
        if len(name1)>3 and len(name2)>3:
            num1 = (int(name1[2])-1)*3+int(name1[3])+1
            num11 = (int(name1[0])-1)*10+int(name1[1])
            angle1= self.angle_refine_uwa[num11-1,num1-1][1]
            num2 =  (int(name2[2])-1)*3+int(name2[3])+1
            num22 =  (int(name2[0])-1)*10+int(name2[1])
            angle2= self.angle_refine_uwa[num22-1,num2-1][1]
        elif "CFPose" in self.pairs[item][0] :
            a = int(self.pairs[item][0].split("/")[-1].split('_')[1].split(".")[0])
            b = int(self.pairs[item][2].split("/")[-1].split('_')[1].split(".")[0])
            i = int(self.pairs[item][2].split("/")[-1].split('_')[0])
            angle1= self.angle_refine[i-1,a-1][1]
            angle2= self.angle_refine[i-1,b-1][1]
        mat1 =scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][0]))['minu']
        dis1 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][1]))['feature_p']
        mat2 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][2]))['minu']
        dis2 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][3]))['feature_p']
        label = np.array(self.pairs[item][4]).astype('int64')
        rand = torch.rand(1)
        if rand < 0.5:
            minu1 = mat1[:, :6].astype('float32')
            disc1 = dis1[:, :].astype('float32')
            minu2 = mat2[:, :6].astype('float32')
            disc2 = dis2[:, :].astype('float32')
        else:
            minu1 = mat2[:, :6].astype('float32')
            disc1 = dis2[:, :].astype('float32')
            minu2 = mat1[:, :6].astype('float32')
            disc2 = dis1[:, :].astype('float32')

        minu1[:, :3] -= (np.mean(minu1[:, :3], axis=0))
        minu2[:, :3] -= (np.mean(minu2[:, :3], axis=0))
        minu1 = np.nan_to_num(minu1)
        minu2 = np.nan_to_num(minu2)
        disc1 = np.nan_to_num(disc1)
        disc2 = np.nan_to_num(disc2)
        # 随机旋转
        if self.training:
            if len(minu1) > 50:
                num =  int(np.random.uniform(0, 110))
                if num < 0:
                    num = 1
                if num > len(minu1):
                    num = int(len(minu1) / 4)
                changes = np.random.choice(range(0, len(minu1)), size=num, replace=False)
                minu1[changes] = np.zeros((num, 6))
                disc1[changes] = np.zeros((num, 128))
            if len(minu2) > 50:
                num =  int(np.random.uniform(0, 110))
                if num < 0:
                    num = 1
                if num > len(minu2):
                    num = int(len(minu2) / 4)
                changes = np.random.choice(range(0, len(minu2)), size=num, replace=False)
                minu2[changes] = np.zeros((num, 6))
                disc2[changes] = np.zeros((num, 128))


        if len(minu1) >= self.num_points:
            minu1 = minu1[:self.num_points]
            disc1 = disc1[:self.num_points]
        else:
            minu1 = np.concatenate((minu1,np.zeros((self.num_points-len(minu1),6))))
            disc1 = np.concatenate((disc1,np.zeros((self.num_points-len(disc1),128))))
        if len(minu2) >= self.num_points:
            minu2 = minu2[:self.num_points]
            disc2 = disc2[:self.num_points]
        else:
            minu2 = np.concatenate((minu2, np.zeros((self.num_points - len(minu2), 6))))
            disc2 = np.concatenate((disc2,np.zeros((self.num_points-len(disc2),128))))
        np.random.shuffle(minu1)
        np.random.shuffle(minu2)
        return minu1,disc1, minu2,disc2, label,angle1, angle2

    def __len__(self):
        return len(self.pairs)

    def _read_data(self):
        pairs = []
        if self.training:
            with open('pair_train.txt','r') as f:
                for line in f:
                    file1,disc1, file2,disc2, label = line.split(" ")
                    label =int(label)
                    pair = (file1,disc1, file2,disc2,label)
                    pairs.append(pair)
        else:
            with open('pair_test.txt', 'r') as f:
                for line in f:
                    file1,disc1, file2,disc2, label = line.split(" ")
                    label = int(label)
                    pair = (file1,disc1, file2,disc2, label)
                    pairs.append(pair)
            pairs= pairs[::-1]
        return pairs

class FingerLoader_Identify(Dataset):
    def __init__(self, num_points, img_dir,data1=True,uwa=True,img_dir2=None):
        self.BASE_DIR = img_dir
        self.BASE_DIR2 = img_dir if img_dir2 is None else img_dir2
        self.data1 = data1
        self.uwa= uwa

        self.pairs = self._read_data()
        self.num_points = num_points

    def __getitem__(self, item):
        mat1 =scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][0]))['minu']
        dis1 = scipy.io.loadmat(os.path.join(self.BASE_DIR2,self.pairs[item][1]))['feature_p']
        label = np.array(self.pairs[item][2]).astype('int64')

        minu1 = mat1[:, :6].astype('float32')
        disc1 = dis1[:, :].astype('float32')


        minu1[:, :3] -= (np.mean(minu1[:, :3], axis=0))


        if len(minu1) >= self.num_points:
            minu1 = minu1[:self.num_points]
            disc1 = disc1[:self.num_points]
        else:
            minu1 = np.concatenate((minu1,np.zeros((self.num_points-len(minu1),6))))
            disc1 = np.concatenate((disc1,np.zeros((self.num_points-len(disc1),128))))
        return minu1,disc1, label,self.pairs[item][0]

    def __len__(self):
        return len(self.pairs)

    def _read_data(self):
        pairs = []
        if self.uwa==True:
            if self.data1:
                with open('pair_identify_uwa1.txt','r') as f:
                    for line in f:
                        file1,disc1, label = line.split(" ")
                        # disc1 = "/mnt/sdc/jyw_data/3d_2d/DMD/DMD_disc/" + disc1.split('/')[-1].split(".")[0] + ".npy"
                        label =int(label)
                        pair = (file1,disc1,label)
                        pairs.append(pair)
            else:
                with open('pair_identify_uwa2.txt','r') as f:
                    for line in f:
                        file1,disc1, label = line.split(" ")
                        # disc1 = "/mnt/sdc/jyw_data/3d_2d/DMD/DMD_disc/" + disc1.split('/')[-1].split(".")[0] + ".npy"
                        label =int(label)
                        pair = (file1,disc1,label)
                        pairs.append(pair)
        else:
            with open('pair_identify_cfpose.txt', 'r') as f:
                for line in f:
                    file1, label = line.split(" ")
                    disc1 = os.path.join(self.BASE_DIR2,file1)

                    file1 = os.path.join(self.BASE_DIR,file1.split(".")[0])

                    label = int(label)
                    pair = (file1, disc1, label)
                    pairs.append(pair)

        return pairs
class FingerLoader_Identify_ZJU(Dataset):
    def __init__(self, num_points, img_dir,data1=True):
        self.BASE_DIR = img_dir
        self.data1 = data1
        self.angle_refine = scipy.io.loadmat("/mnt/sdb/CFPose/core/my_angle_refine_all.mat")['angle_refine']

        self.pairs = self._read_data()
        self.num_points = num_points

    def __getitem__(self, item):
        mat1 =scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][0]))['minu']
        dis1 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][1]))['feature_p']
        mat2 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][2]))['minu']
        dis2 = scipy.io.loadmat(os.path.join(self.BASE_DIR,self.pairs[item][3]))['feature_p']
        rand = torch.rand(1)
        label = np.array(self.pairs[item][4]).astype('int64')

        if rand < 0.5:
            minu1 = mat1[:, :6].astype('float32')
            disc1 = dis1[:, :].astype('float32')
            minu2 = mat2[:, :6].astype('float32')
            disc2 = dis2[:, :].astype('float32')
        else:
            minu1 = mat2[:, :6].astype('float32')
            disc1 = dis2[:, :].astype('float32')
            minu2 = mat1[:, :6].astype('float32')
            disc2 = dis1[:, :].astype('float32')

        minu1[:, :3] -= (np.mean(minu1[:, :3], axis=0))
        minu2[:, :3] -= (np.mean(minu2[:, :3], axis=0))
        minu1 = np.nan_to_num(minu1)
        minu2 = np.nan_to_num(minu2)
        disc1 = np.nan_to_num(disc1)
        disc2 = np.nan_to_num(disc2)


        if len(minu1) >= self.num_points:
            minu1 = minu1[:self.num_points]
            disc1 = disc1[:self.num_points]
        else:
            minu1 = np.concatenate((minu1,np.zeros((self.num_points-len(minu1),6))))
            disc1 = np.concatenate((disc1,np.zeros((self.num_points-len(disc1),128))))
        if len(minu2) >= self.num_points:
            minu2 = minu2[:self.num_points]
            disc2 = disc2[:self.num_points]
        else:
            minu2 = np.concatenate((minu2, np.zeros((self.num_points - len(minu2), 6))))
            disc2 = np.concatenate((disc2,np.zeros((self.num_points-len(disc2),128))))
        np.random.shuffle(minu1)
        np.random.shuffle(minu2)
        return minu1,disc1, minu2,disc2,label

    def __len__(self):
        return len(self.pairs)

    def _read_data(self):
        pairs = []
        if self.data1:
            with open('pair_identify_zju_true.txt','r') as f:
                for line in f:
                    file1,disc1, file2,disc2,label = line.split(" ")
                    label =int(label)
                    pair = (file1,disc1, file2,disc2,label)
                    pairs.append(pair)
        else:
            with open('pair_identify_zju_false.txt', 'r') as f:
                for line in f:
                    file1,disc1, file2,disc2,label = line.split(" ")
                    label = int(label)
                    pair = (file1,disc1, file2,disc2,label)
                    pairs.append(pair)
        return pairs


class FingerLoader_pretrain(Dataset):
    def __init__(self, num_points, img_dir,training=True):
        self.BASE_DIR = img_dir
        self.training = training
        self.angle_refine = scipy.io.loadmat("/mnt/sdb/CFPose/core/my_angle_refine_all.mat")['angle_refine']

        self.pairs = self._read_data()
        self.num_points = num_points



    def __getitem__(self, item):
        mat1 =np.load(os.path.join(self.BASE_DIR,self.pairs[item][0]))
        dis1 = np.load(os.path.join(self.BASE_DIR,self.pairs[item][1]))
        mat2 = np.load(os.path.join(self.BASE_DIR,self.pairs[item][2]))
        dis2 = np.load(os.path.join(self.BASE_DIR,self.pairs[item][3]))
        label = np.array(self.pairs[item][4]).astype('int64')
        rand = torch.rand(1)
        if rand < 0.5:
            minu1 = mat1[:, :].astype('float32')
            disc1 = dis1[:, :].astype('float32')
            minu2 = mat2[:, :].astype('float32')
            disc2 = dis2[:, :].astype('float32')
        else:
            minu1 = mat2[:, :].astype('float32')
            disc1 = dis2[:, :].astype('float32')
            minu2 = mat1[:, :].astype('float32')
            disc2 = dis1[:, :].astype('float32')

        if minu1.shape[1]<6:
            minu1 = from_2d_to_3d(minu1)
            minu2 = from_2d_to_3d(minu2)
        minu1[:, :3] -= (np.mean(minu1[:, :3], axis=0))
        minu2[:, :3] -= (np.mean(minu2[:, :3], axis=0))
        # 随机旋转
        if self.training:
            # if num1 == num2:
            rot1,theta1,_,_ = get_3d_rotate_mat()
            new_pos = np.dot(rot1, minu1[:, :3].T).T
            new_theta = np.dot(rot1, minu1[:, 3:6].T).T
            minu1 = np.concatenate((new_pos, new_theta), axis=1)

            rot2,theta2,_,_ = get_3d_rotate_mat()

            new_pos = np.dot(rot2, minu2[:, :3].T).T
            new_theta = np.dot(rot2, minu2[:, 3:6].T).T
            minu2 = np.concatenate((new_pos, new_theta), axis=1)
            if len(minu1) > 20:
                num = int(len(minu1) / 6) + int(10 * np.random.normal(size=1))
                if num < 0:
                    num = 1
                if num > len(minu1):
                    num = int(len(minu1) / 6)
                minu1[np.random.choice(range(0, len(minu1)), size=num, replace=False)] = np.zeros((num, 6))
                disc1[np.random.choice(range(0, len(disc1)), size=num, replace=False)] = np.zeros((num, 768))
            if len(minu2) > 20:
                num = int(len(minu2) / 6) + int(10 * np.random.normal(size=1))
                if num < 0:
                    num = 1
                if num > len(minu2):
                    num = int(len(minu2) / 6)
                minu2[np.random.choice(range(0, len(minu2)), size=num, replace=False)] = np.zeros((num, 6))
                disc2[np.random.choice(range(0, len(disc2)), size=num, replace=False)] = np.zeros((num, 768))

        if len(minu1) >= self.num_points:
            minu1 = minu1[:self.num_points]
            disc1 = disc1[:self.num_points]
        else:
            minu1 = np.concatenate((minu1,np.zeros((self.num_points-len(minu1),6))))
            disc1 = np.concatenate((disc1,np.zeros((self.num_points-len(disc1),768))))
        if len(minu2) >= self.num_points:
            minu2 = minu2[:self.num_points]
            disc2 = disc2[:self.num_points]
        else:
            disc2 = np.concatenate((disc2,np.zeros((self.num_points-len(disc2),768))))
            minu2 = np.concatenate((minu2, np.zeros((self.num_points - len(minu2), 6))))

            np.random.shuffle(minu1)
        np.random.shuffle(minu2)
        return minu1,disc1, minu2,disc2, label

    def __len__(self):
        return len(self.pairs)

    def _read_data(self):
        pairs = []
        if self.training:
            with open('pair_dmd_pretrain.txt','r') as f:
                for line in f:
                    file1,disc1, file2,disc2, label = line.split(" ")
                    label =int(label)
                    pair = (file1,disc1, file2,disc2,label)
                    pairs.append(pair)
        else:
            with open('pair_dmd_pretrain_test.txt', 'r') as f:
                for line in f:
                    file1,disc1, file2,disc2, label = line.split(" ")
                    label = int(label)
                    pair = (file1,disc1, file2,disc2, label)
                    pairs.append(pair)
        return pairs
