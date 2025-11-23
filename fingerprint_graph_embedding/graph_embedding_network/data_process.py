import os
import scipy.io
from matplotlib import pyplot as plt
from PIL import Image
import  numpy as np

def find_middle(z):
    max_z = -np.inf
    min_z = np.inf
    for i in z:
        for j in i:
            if j!=0 and j>max_z:
                max_z = j
    for i in z:
        for j in i:
            if j!=0 and j<min_z:
                min_z = j
    return (max_z+min_z)/2

def get_local_direction(pos,angle,length = 25):
    """绘制细节点方向"""
    out = []
    for i in range(pos.shape[0]):
        dx = np.cos(angle[i][0]/180*np.pi)
        dy = np.sin(angle[i][0]/180*np.pi)
        len1 = np.sqrt(angle[i][1]**2+1)
        len2 = np.sqrt(angle[i][2]**2+1)
        dz = np.cos(angle[i][0]/180*np.pi)*angle[i][1]/len1+np.sin(angle[i][0]/180*np.pi)*angle[i][2]/len2
        len3 = length/np.sqrt(dx**2+dy**2+dz**2)
        rotate = np.array((dx*len3,dy*len3,dz*len3))
        if np.isnan(rotate).any():
            print(rotate)
            out.append(np.array((0,0,0)))
        else:
            out.append(rotate)

    return np.array(out)

def get_minu_3d(mat,mat2):
    minus = mat2['df_p']
    minus[:,:2] = (minus[:,:2]).astype(int)
    len_minu = len(minus)
    minus_3d = []
    z_all = mat['depth_p']
    fy_all, fx_all = np.gradient(z_all)

    for i in range(len_minu):
        pos = mat2['df_p'][i][0:2].astype(int)
        z = z_all[pos[1], pos[0]]
        fx = fx_all[pos[1], pos[0]]
        fy = fy_all[pos[1], pos[0]]
        rx = np.arctan(fx) / np.pi * 180
        ry = np.arctan(fy) / np.pi * 180
        minu_3d = [minus[i][0], minus[i][1], z, minus[i][2], rx, ry]
        minus_3d.append(minu_3d)
    minus_3d = np.array(minus_3d)

    h, w = z_all.shape
    mid_z = find_middle(z_all)
    print(mid_z)
    minus_3d = minus_3d - np.array([w / 2, h / 2, mid_z, 0, 0, 0])

    direction = get_local_direction(minus_3d[:, :3], minus_3d[:, 3:6])
    minus_3d = np.concatenate((minus_3d[:, :3], direction), axis=1)
    return minus_3d

def get_minus_3d_uwa():
    """
    BASE_DIR里存深度，BASE_DIR2存2D细节点，输出3D细节点到OUT_DIR
    """
    BASE_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_pred"
    BASE_DIR2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_feature"
    OUT_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_df_3d"
    files = os.listdir(BASE_DIR2)
    for file1 in files:
        # try:
        if not os.path.exists(os.path.join(OUT_DIR,file1.split(".")[0])):
            mat1 = scipy.io.loadmat(os.path.join(BASE_DIR,file1))
            mat1_1 = scipy.io.loadmat(os.path.join(BASE_DIR2,file1))
            try:
                minus_3d = get_minu_3d(mat1,mat1_1)
            except:
                print("no minu! "+str(file1))
                continue
            scipy.io.savemat(os.path.join(OUT_DIR,file1.split(".")[0]),mdict={'minu':minus_3d})
            print(os.path.join(OUT_DIR,file1.split(".")[0]))

def get_minus_3d_cfpose():
    """
    BASE_DIR里存深度，BASE_DIR2存2D细节点，输出3D细节点到OUT_DIR
    """
    BASE_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_pred"
    BASE_DIR2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_feature"
    OUT_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_df_3d"


    files = os.listdir(BASE_DIR2)
    for file1 in files:
        # try:
        # if not os.path.exists(os.path.join(OUT_DIR,file1.split(".")[0])):
        mat1 = scipy.io.loadmat(os.path.join(BASE_DIR,file1))
        mat1_1 = scipy.io.loadmat(os.path.join(BASE_DIR2,file1))
        minus_3d = get_minu_3d(mat1,mat1_1)
        scipy.io.savemat(os.path.join(OUT_DIR,file1.split(".")[0]),mdict={'minu':minus_3d})
        print(os.path.join(OUT_DIR,file1.split(".")[0]))
        # except:
        #     print(file1 + " failed")

def get_minus_3d_zju():
    """
    BASE_DIR里存深度，BASE_DIR2存2D细节点，输出3D细节点到OUT_DIR
    """
    BASE_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_pred"
    BASE_DIR2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_feature"
    OUT_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_df_3d"

    files = os.listdir(BASE_DIR2)
    for file1 in files:
        files2 = os.listdir(os.path.join(BASE_DIR2, file1))

        for file2 in files2:
            if not os.path.exists(os.path.join(OUT_DIR, file1)):
                os.mkdir(os.path.join(OUT_DIR, file1))
            # try:
            if not os.path.exists(os.path.join(OUT_DIR, file1, file2.split(".")[0])):
                mat1 = scipy.io.loadmat(os.path.join(BASE_DIR, file1, file2))
                mat1_1 = scipy.io.loadmat(os.path.join(BASE_DIR2, file1, file2))
                try:
                    minus_3d = get_minu_3d(mat1, mat1_1)
                except:
                    print("no minu! " + str(file1))
                    continue
                scipy.io.savemat(os.path.join(OUT_DIR, file1, file2.split(".")[0]), mdict={'minu': minus_3d})
                print(os.path.join(OUT_DIR, file1, file2.split(".")[0]))

def generate_pairs_train():
    """
    构建用于训练三元组损失的样本对
    """
    BASE_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_df_3d"
    BASE_DIR2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_feature"
    with open('pair_train.txt','w') as f:
        with open('pair_test.txt','w') as f2:
            pairs = []
            for i in range(1,151):
                for j in range(1,11):
                    for a in range(1,3):
                        for b in range(0,3):
                            for a1 in range(1,3):
                                for b1 in range(0,3):
                                    if a1 != a or b1 !=b:
                                        file1 = str(i)+"_"+str(j)+"_"+str(a)+"_"+str(b)
                                        file11 = os.path.join(BASE_DIR2,file1)
                                        file1 = os.path.join(BASE_DIR,file1)
                                        file2 = str(i)+"_"+str(j)+"_"+str(a1)+"_"+str(b1)
                                        file21 = os.path.join(BASE_DIR2,file2)

                                        file2 = os.path.join(BASE_DIR,file2)


                                        label = str((i-1)*10+(j))
                                        pair = (file1,file11,file2,file21,label)
                                        if (file2,file1,label) not in pairs:
                                            pairs.append(pair)
                                            if os.path.exists(file1) and os.path.exists(file2):
                                                if i>100:
                                                    if b==b1:
                                                        nums = 4
                                                    elif b==1 and b1==2:
                                                        nums=1
                                                    else:
                                                        nums=2

                                                    label2 = str(int(label) - 1000)
                                                    pair = (file1,file11,file2,file21,label2)
                                                    for _ in range(nums):
                                                        f.write(" ".join(pair) + "\n")
                                                else:
                                                    f2.write(" ".join(pair) + "\n")
            pairs = []
            for i in range(1,141):
                for a in range(1,11):
                    for b in range(a+1,11):
                        file1 = str(i) + "_" + str(a)
                        file11 = os.path.join(BASE_DIR2, file1)

                        file1 = os.path.join(BASE_DIR, file1)

                        file2 = str(i) + "_" + str(b)
                        file21 = os.path.join(BASE_DIR2, file2)

                        file2 = os.path.join(BASE_DIR, file2)

                        label = str(i)
                        pair = (file1,file11, file2,file21, label)
                        if (file2, file1, label) not in pairs:
                            pairs.append(pair)
                            if os.path.exists(file1) and os.path.exists(file2):
                                if i > 120:
                                    label = str(i-120+500)
                                    pair = (file1,file11, file2,file21, label)
                                    for _ in range(2):
                                        f.write(" ".join(pair) + "\n")
                                else:
                                    label = str(i+1000)
                                    pair = (file1,file11, file2,file21, label)

                                    f2.write(" ".join(pair) + "\n")

def generate_pairs_identify_uwa():
    """
    构建用于测试的样本
    """
    path = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_df_3d"
    path2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_feature"
    with open('pair_identify_all_uwa1.txt','w') as f:
        with open('pair_identify_all_uwa2.txt','w') as f2:
            for i in range(1,101):
                for j in range(1,11):
                    for k in range(3):
                        for l in range(1,3):
                            file1 = str(i)+"_"+str(j)+"_"+str(l)+"_"+str(k)
                            label = str((i-1) * 10 + (j - 1))
                            flag1 = False
                            minu1 = os.path.join(path, file1)
                            disc1 = os.path.join(path2, file1 + ".mat")
                            # if file1 not in remove and file2 not in remove:
                            try:
                                mat1 = scipy.io.loadmat(minu1)
                                mat2 = scipy.io.loadmat(disc1)
                                flag1 =True
                            except:
                                print("no file:" + file1)
                            if os.path.exists(minu1)  and os.path.exists(disc1):
                                if l==1:
                                    f.write(minu1 + ' ' + disc1 + ' ' + label+"\n")
                                else:
                                    f2.write(minu1 + ' ' + disc1 + ' ' + label+"\n")

def generate_pairs_identify_cfpose():
    BASE_DIR = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_CFPose_df_3d"
    with open('pair_identify_cfpose.txt','w') as f:
        for i in range(1,121):
            for j in range(1,11):
                file1 = str(i)+"_"+str(j)+".mat"
                label = str((i-1))
                flag1 = False
                try:
                    mat1 = scipy.io.loadmat(os.path.join(BASE_DIR, file1))
                    flag1 =True
                except:
                    print("no file:" + file1)
                if flag1:
                    f.write(
                        file1+" "+ label+"\n")

def generate_pairs_identify_ZJU():
    """
    构建用于测试的样本
    """
    path = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_df_3d"
    path2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_feature"
    out = sorted(os.listdir(path))
    dict_out = {name: idx for idx, name in enumerate(out)}
    with open('pair_identify_all_zju1.txt','w') as f:
        with open('pair_identify_all_zju2.txt','w') as f2:
            for name,idx in dict_out.items():
                files = sorted(os.listdir(os.path.join(path2,name)))
                for file in files:
                    file1 = file.split(".")[0]
                    dict2 = {"L1":1,"L2":2,"R1":3,"R2":4}
                    label = idx*4 + dict2[file1[:2]]
                    flag1 = False
                    minu1 = os.path.join(path, name,file1 )
                    disc1 = os.path.join(path2,name, file1+ ".mat")
                    # if file1 not in remove and file2 not in remove:
                    try:
                        mat1 = scipy.io.loadmat(minu1)
                        mat2 = scipy.io.loadmat(disc1)
                        flag1 =True
                    except:
                        print("no file:" + file1)
                    if os.path.exists(minu1)  and os.path.exists(disc1):
                        if int(file1.split("_")[1]) == 1:
                            f.write(minu1 + ' ' + disc1 + ' ' +str(label) + "\n")
                        else:
                            f2.write(minu1 + ' ' + disc1 + ' ' + str(label) + "\n")

def generate_pairs_identify_ZJU_det():
    """
    构建用于测试的样本
    """
    path = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_df_3d"
    path2 = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_ZJU_feature"
    out = sorted(os.listdir(path))
    out_list0 = {}

    out_list = {}

    dict_out = {name: idx for idx, name in enumerate(out)}
    with open('pair_identify_zju_true.txt','w') as f:
        with open('pair_identify_zju_false.txt','w') as f2:
            for name,idx in dict_out.items():
                files = sorted(os.listdir(os.path.join(path,name)))
                for file in files:
                    for file2 in files:
                        file1 = file
                        file2 = file2
                        if file1!=file2:
                            dict2 = {"L1":1,"L2":2,"R1":3,"R2":4}
                            label = idx*4 + dict2[file1[:2]]
                            label2 = idx*4 + dict2[file2[:2]]
                            if label==label2:
                                flag1 = False
                                minu1 = os.path.join(path, name,file1)
                                disc1 = os.path.join(path2,name, file1 + ".mat")
                                minu2 = os.path.join(path, name,file2)
                                disc2 = os.path.join(path2,name, file2 + ".mat")
                                # if file1 not in remove and file2 not in remove:
                                # try:
                                #     mat1 = scipy.io.loadmat(minu1)
                                #     mat2 = scipy.io.loadmat(disc1)
                                #     flag1 =True
                                 # except:
                                #     print("no file:" + file1)
                                out_list0[minu1+"+"+minu2]=True
                                if os.path.exists(minu1)  and os.path.exists(disc1) and os.path.exists(minu2)  and os.path.exists(disc2):
                                    # if file1[1] == "1":
                                    if minu2+"+"+minu1 not in out_list0:

                                        f.write(minu1 + ' ' + disc1 + ' '+minu2 + ' ' + disc2+' '+str(label)+"\n")
            for name, idx in dict_out.items():
                files = sorted(os.listdir(os.path.join(path,name)))
                for file in files:
                    for name2, idx2 in dict_out.items():
                        files2 = sorted(os.listdir(os.path.join(path, name2)))
                        for file2 in files2:
                            file1 = file

                            if int(file1.split("_")[1]) == 1 and int(file2.split("_")[1]) == 1:

                                dict2 = {"L1":1,"L2":2,"R1":3,"R2":4}
                                label = idx*4 + dict2[file1[:2]]
                                flag1 = False
                                minu1 = os.path.join(path, name,file1)
                                disc1 = os.path.join(path2,name, file1 + ".mat")
                                label2 = idx2*4 + dict2[file2[:2]]

                                minu2 = os.path.join(path, name2,file2)
                                disc2 = os.path.join(path2,name2, file2 + ".mat")
                                # if file1 not in remove and file2 not in remove:
                                # try:
                                #     mat1 = scipy.io.loadmat(minu1)
                                #     mat2 = scipy.io.loadmat(disc1)
                                #     flag1 =True
                                # except:
                                #     print("no file:" + file1)
                                out_list[minu1+"+"+minu2]=True

                                if os.path.exists(minu1)  and os.path.exists(disc1)  and os.path.exists(minu2)  and os.path.exists(disc2):
                                    if label!=label2:
                                        if minu2+"+"+minu1 not in out_list:
                                            f2.write(minu1 + ' ' + disc1 +' '+ minu2 + ' ' + disc2 +' ' + str(label)+"\n")


# get_minus_3d_uwa()
# get_minus_3d_cfpose()
# get_minus_3d_zju()
generate_pairs_train()
generate_pairs_identify_uwa()
generate_pairs_identify_cfpose()
generate_pairs_identify_ZJU_det()
