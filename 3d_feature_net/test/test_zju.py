import cv2
import numpy as np
import os

from scipy.io import savemat, loadmat

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn

from models.backbone.depth_encoder import LightMonoAll  # 假设 model1.py 已经被翻译成 PyTorch
from models.utils.utils import locate_minutiae
from scipy import ndimage
HEIGHT = 960
WIDTH = 720
base_dir = "/mnt/sdc/jyw_data/3d_2d/raw_data_ZJU"
# save_dir = "/mnt/sdc/jyw_data/3d_2d/raw_data_ZJU_pred"
save_dir = "/mnt/sdc/jyw_data/3d_2d/raw_data_640_pred_test"

epoch_num = 1
torch.manual_seed(3047)
def locate_minutiae2(minu_out,ori_out,tau_init=0.4,tau_step=0.4,t_area=30):
    _, minu_pred = locate_minutiae(minu_out.squeeze(1),tau_init=tau_init,tau_step=tau_step,t_area=t_area)
    minu_pred2 = []
    ori_pred = torch.argmax(ori_out.squeeze(), dim=0).float().cpu().numpy()
    ori_pred = ndimage.zoom(np.squeeze(ori_pred), [4, 4], order=3)
    for minu in minu_pred:
        minu_pred2.append([int(minu[0] / 1.5), int(minu[1] / 1.5), int(-ori_pred[minu[1], minu[0]] + 90)])
    minu_pred2 = np.array(minu_pred2)
    return minu_pred2

def equalize_img(img,mask):
    h, w = img.shape[:2]
    noises = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
    img=img*mask+noises*(1-mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img.astype(np.uint8))
    img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    return img

def main():

    # 初始化模型、优化器和学习率调度器
    model = LightMonoAll()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("ckpt/model_3d_feature.pth"))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scale = 4
    down_sample = scale*4
    files = os.listdir(base_dir)
    for i in range(len(files)):
        files2 = os.listdir(os.path.join(base_dir,files[i]))
        files2 = [f for f in files2 if ".mat" in f]

        for f2 in files2:
            if not os.path.exists(os.path.join(save_dir, files[i])):
                path_img = os.path.join(base_dir,files[i],f2)
                img = loadmat(path_img)['img']
                mask = loadmat(path_img)['mask']
                img = cv2.resize(img,(WIDTH,HEIGHT))
                mask = cv2.resize(mask,(WIDTH,HEIGHT))
                img = torch.tensor(img).unsqueeze(0).unsqueeze(0)/255
                mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

                # img_input = img_input.to(device).float()
                if WIDTH % down_sample != 0 or HEIGHT % down_sample != 0:
                    ww = down_sample * (int(WIDTH / down_sample) + 1)
                    hh = down_sample * (int(HEIGHT / down_sample) + 1)
                    img_p = torch.zeros((1, 1, hh, ww), dtype=torch.float32).to(device)
                    mask_p = torch.zeros((1, 1, hh, ww), dtype=torch.float32).to(device)
                    img_p[:, :, :HEIGHT, :WIDTH] = img
                    mask_p[:, :, :HEIGHT, :WIDTH] = mask
                    img = img_p
                    mask = mask_p
                else:
                    img_p = torch.zeros((1, 1, HEIGHT, WIDTH), dtype=torch.float32).to(device)
                    mask_p = torch.zeros((1, 1, HEIGHT, WIDTH), dtype=torch.float32).to(device)
                    img_p[:, :, :HEIGHT, :WIDTH] = img
                    mask_p[:, :, :HEIGHT, :WIDTH] = mask
                    img = img_p
                    mask = mask_p

                img_input = torch.cat([img, img, img], dim=1).to(device).float()

                # 前向传播
                ori_out, ped_out, grad_out,minu_out,feature = model(img_input)
                minu_pred2 = locate_minutiae2(minu_out, ori_out)
                minu_pred3 = locate_minutiae2(minu_out*mask,ori_out,tau_init=0.2,tau_step=0.2)

                grad_out = grad_out.permute(0, 2, 3, 1)
                grad = ndimage.zoom(np.squeeze(grad_out.cpu().detach().numpy()), [scale/3*2, scale/3*2, 1], order=1)

                if not os.path.exists(os.path.join(save_dir,files[i])):
                    os.mkdir(os.path.join(save_dir,files[i]))
                f = os.path.join(save_dir, files[i],f2)
                out = []
                for minu in minu_pred3:
                    out.append(feature[0,:,minu[1],minu[0]].cpu().detach().numpy())
                out = np.array(out)
                savemat(f, {'grad_p': grad,'minu_p':minu_pred2,'df_p':minu_pred3, 'feature_p': out})
                print(str(i)+" "+ files[i]+" "+f2)


if __name__ == "__main__":
    main()
