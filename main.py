import torch
from nets import FLM, Dt, Dv, G
import matplotlib.pyplot as plt
import torch.optim as optim
import load_data
from scipy.spatial.distance import cdist
import h5py
import numpy as np
import scipy.io as scio
from PIL import Image
from train import train_model
from utils import CalcTopMap, optimized_mAP
import sys
import random as sh

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


# def pr_curve(qB, rB, qL, rL, ep, task,  topK=-1)


def pr_curve(qB, rB, qL, rL, topK=600):
    n_query = qB.shape[0]
    print("n_query", n_query)
    if topK == -1 or topK > rB.shape[0]:  # top-K 之 K 的上限
        topK = rB.shape[0]

    # Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)

    Gnd = (qL.mm(rL.transpose(0, 1)) > 0).type(torch.float32)
    _, Rank = torch.sort(calc_hammingDist(qB, rB))
    P, R = [], []
    # KK = []
    # K_ = [x * 2000 + 1 for x in range(1, int(topK/2000))]
    # for i in K_:
    #     if i < topK:
    #         KK.append(i)
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        print(k)
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample

            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue

            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(torch.mean(p))
        R.append(torch.mean(r))
    print(P)
    print(R)
    return P, R


# 画 P-R 曲线


def guiyi(x):

    x = (x - np.amin(x, 0) / 2 - np.amax(x, 0) / 2) / (np.amax(x, 0) - np.amin(x, 0))
    return x


def linjie(l1, l2):
    l2 = l2.t()

    adj = torch.matmul(l1, l2)
    one = torch.ones_like(adj)
    zero = torch.zeros_like(adj)
    return torch.where(adj > 0, one, zero)

    return adj


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    torch.manual_seed(1)
    np.random.seed(1)
    dataset = 'pascal'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-1
    beta = 1e-1
    MAX_EPOCH = 20
    batch_size = 20
    # batch_size = 512
    lr = 1e-8
    LR_G = 0.00001
    LR_Dt = 0.00001
    LR_Dv = 0.00001
    betas = (0.5, 0.999)
    weight_decay = 0

    print('...Data loading is beginning...')

    label_set = scio.loadmat('./mirflickr/mirflickr25k-lall.mat')
    label_set = np.array(label_set['LAll'], dtype=np.float64)
    txt_set = scio.loadmat('./mirflickr/mirflickr25k-yall.mat')
    txt_set = np.array(txt_set['YAll'], dtype=np.float64)
    mirflickr = h5py.File('./mirflickr/mirflickr25k-iall.mat', 'r', libver='latest', swmr=True)

    images = mirflickr['IAll'][:].transpose(0, 1, 2, 3)

    mirflickr.close()

    print(images.shape)
    print(txt_set.shape)

    print(label_set.shape)




    images_train = images[2000:]
    images_test = images[:2000]
    txt_train = txt_set[2000:]
    txt_test = txt_set[:2000]
    label_train = label_set[2000:]
    label_test = label_set[:2000]
    """
    images_train = images[20:200]
    images_test = images[:20]
    txt_train = txt_set[20:200]
    txt_test = txt_set[:20]
    label_train = label_set[20:200]
    label_test = label_set[:20]
    """





    # [63 32 80 33 61 45 28 55 39 80]

    b = label_train[np.random.choice(label_train.shape[0],128),:]
    print(b.shape)







    data_loader, input_data_par = load_data.get_loader(images_train, images_test, txt_train, txt_test, label_train,
                                                       label_test, batch_size)
    print('...Training is completed...')

    print('...Evaluation on testing data...')

    model_ft = FLM().to(device)
    G = G().to(device)
    Dt = Dt().to(device)
    Dv = Dv().to(device)
    params_to_update = list(model_ft.parameters())
    Gparams_to_update = list(G.parameters())
    Dparams_to_update = list(Dt.parameters())
    Dparams_to_update1 = list(Dv.parameters())


    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    optimizer_G = torch.optim.Adam(Gparams_to_update, lr=LR_G)
    optimizer_Dt = torch.optim.Adam(Dparams_to_update, lr=LR_Dt)
    optimizer_Dv = torch.optim.Adam(Dparams_to_update1, lr=LR_Dv)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, model_G, model_Dt, model_Dv, hashing = train_model(model_ft, G, Dt, Dv, data_loader, optimizer, optimizer_G,
                                                                   optimizer_Dt, optimizer_Dv, alpha,
                                                                   beta, num_epochs=MAX_EPOCH, key1=b)
    print('...Training is completed...')

    print('...Evaluation on testing data...')



    x, x1, x2, x5, y, y1, y2, y5 = model_G(
        torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = torch.tensor(input_data_par['label_test'])
    """
    x0, x10, x20, x50, y0, y10, y20, y50 = model_G(
        torch.tensor(input_data_par['img_train']).to(device), torch.tensor(input_data_par['text_train']).to(device))

    """
    label0 = torch.tensor(input_data_par['label_train'])
    simi_matrix = linjie(label0, label)
    simi_matrix = simi_matrix.detach().cpu().numpy()
    simi_matrix = simi_matrix.astype(int)
    x11, y11, x61, y61, l11, l21 = model_ft(x1, y1, b, 1)
    x21, y21, x62, y62, l12, l22 = model_ft(x1, y1, b, 2)
    # x0, y0, x30, y30, x40, y40, x60, y60 = model_ft(x10, y10)

    # qu_BI = x3.detach().cpu().numpy()
    # qu_BT = y3.detach().cpu().numpy()
    # re_BI = y3.detach().cpu().numpy()
    # re_BT = x3.detach().cpu().numpy()
"""
    qu_BI = hashing_t
    qu_BT = hashing_v
    re_BT = y.detach().cpu().numpy()
    re_BI = x.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    label0 = label0.detach().cpu().numpy()

    qu_BI = guiyi(qu_BI)
    qu_BT = guiyi(qu_BT)
    re_BT = guiyi(re_BT)
    re_BI = guiyi(re_BI)

    # '''

    qu_BI = np.where(qu_BI > 0, qu_BI, -1)
    qu_BI = np.where(qu_BI < 0, qu_BI, 1).astype(int)

    qu_BT = np.where(qu_BT > 0, qu_BT, -1)
    qu_BT = np.where(qu_BT < 0, qu_BT, 1).astype(int)

    re_BT = np.where(re_BT > 0, re_BT, -1)
    re_BT = np.where(re_BT < 0, re_BT, 1).astype(int)

    re_BI = np.where(re_BI > 0, re_BI, -1)
    re_BI = np.where(re_BI < 0, re_BI, 1).astype(int)
    # '''
    label0 = label0.astype(int)
    label = label.astype(int)
    # print(qu_BI)
    # print(qu_BT)
    # print(re_BI)
    # print(re_BT)

    MAP_I2T = optimized_mAP(qu_BI, re_BT, simi_matrix)
    MAP_T2I = optimized_mAP(qu_BT, re_BI, simi_matrix)

    TOPKMAP_I2T = CalcTopMap(qu_BI, re_BT, label0, label)
    TOPKMAP_T2I = CalcTopMap(qu_BT, re_BI, label0, label)

    print('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))

    print('TOPKMAP of Image to Text: %.4f, TOPKMAP of Text to Image: %.4f' % (TOPKMAP_I2T, TOPKMAP_T2I))
"""

qu_B = hashing
re_B1 = x11.detach().cpu().numpy()
re_B2 = x21.detach().cpu().numpy()
label = label.detach().cpu().numpy()
label0 = label0.detach().cpu().numpy()

qu_B = guiyi(qu_B)

re_B1 = guiyi(re_B1)
re_B2 = guiyi(re_B2)



# '''

qu_B = np.where(qu_B > 0, qu_B, -1)
qu_B = np.where(qu_B < 0, qu_B, 1).astype(int)


re_B1 = np.where(re_B1 > 0, re_B1, -1)
re_B1 = np.where(re_B1 < 0, re_B1, 1).astype(int)

re_B2 = np.where(re_B2 > 0, re_B2, -1)
re_B2 = np.where(re_B2 < 0, re_B2, 1).astype(int)

# '''
label0 = label0.astype(int)
label = label.astype(int)
# print(qu_BI)
# print(qu_BT)
# print(re_BI)
# print(re_BT)
c0 = np.ones((1, 3, 224, 224))
c = np.zeros((1, 1386), dtype=float)

c[0, 7] = 1
c[0, 69] = 1


x, x1, x2, x5, y, y1, y2, y5 = model_G(
        torch.tensor(c0).to(device), torch.tensor(c).to(device))
b_B = x2, y2, x62, y62, l12, l22 = model_ft(y1, y1, b, 2)
b_B = x2.detach().cpu().numpy()
b_B = guiyi(b_B)
b_B = np.where(b_B > 0, b_B, -1)
b_B = np.where(b_B < 0, b_B, 1).astype(int)

zzz = np.dot(b_B, qu_B.T)
zzz = zzz.reshape(-1)
zzz = np.where(zzz == np.max(zzz))

zzz = list(zzz)

zzz = zzz[0]

print(zzz[0:6])


MAP_I2T = optimized_mAP(qu_B, re_B1, simi_matrix)
MAP_T2I = optimized_mAP(qu_B, re_B2, simi_matrix)

TOPKMAP_I2T = CalcTopMap(qu_B, re_B1, label0, label)
TOPKMAP_T2I = CalcTopMap(qu_B, re_B2, label0, label)

print('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))

print('TOPKMAP of Image to Text: %.4f, TOPKMAP of Text to Image: %.4f' % (TOPKMAP_I2T, TOPKMAP_T2I))










"""
    P, R = pr_curve(torch.from_numpy(qu_BI), torch.from_numpy(re_BT), torch.from_numpy(label0), torch.from_numpy(label))

    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P)  # 第一个是 x，第二个是 y
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.show()

    print("end")

    zzz = np.dot(re_BI[1314, :], qu_BT.T)

    print(np.where(zzz == np.max(zzz)))
    
"""