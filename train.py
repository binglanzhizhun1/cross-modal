from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy

import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def linjie(l1):

    l2 = l1.t()

    adj = torch.matmul(l1, l2)
    one = torch.ones_like(adj)
    zero = torch.zeros_like(adj)
    return torch.where(adj > 0, one, zero)

    return adj






def calc_loss(x1, y1, x6, y6, l1, l2, labels, alpha, beta):
    '''
    term1 = ((view1_predict-labels_1.float())**2).sum(1).sqrt().mean() + ((view2_predict-labels_2.float())**2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23
    '''
    S1 = linjie(labels)



    term1 = ((x1 - y1)**2).sum(1).sqrt().mean()
    term2 = (((l1- labels)**2).sum(1).sqrt().mean() + ((l2 - labels)**2).sum(1).sqrt().mean())
    term3 = ((x6 - y6)**2).sum(1).sqrt().mean()
    # term3 = (-torch.matmul(S1, torch.matmul(x6, torch.t(y6)))+torch.log(1 + torch.exp(torch.matmul(x6, torch.t(y6))))).sum(1).mean()
    #term3 = ((x - y)**2).sum(1).sqrt().mean()
    im_loss = term1 + alpha * term2  + beta * term3
    return im_loss


def linjie(l1):
    l1 = l1.float()

    l2 = l1.t()

    adj = torch.matmul(l1, l2)
    one = torch.ones_like(adj)
    zero = torch.zeros_like(adj)
    return torch.where(adj > 0, one, zero)

    return adj


def train_model(model, G, Dt, Dv, data_loaders, optimizer, optimizer_G, optimizer_Dt, optimizer_Dv, alpha, beta, key1, device="cpu", num_epochs=500):


    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss_history =[]

    key1 = torch.tensor(key1)


    hashing_test = []
    lable_test = []
    hashing_test = np.array(hashing_test)
    lable_test = np.array(lable_test)



    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                G.train()
                Dt.train()
                Dv.train()
                model.train()
            else:
                # Set model to evaluate mode
                G.eval()
                Dt.eval()
                Dv.eval()
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                # imgs = imgs.to(device)
                # txts = txts.to(device)
                # labels = labels.to(device)
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        key1 = key1.cuda()



                    # zero the parameter gradients
                    #optimizer.zero_grad()
                    optimizer_G.zero_grad()
                    optimizer_Dt.zero_grad()
                    optimizer_Dv.zero_grad()

                    # Forward
                    x, x1, x2, x5, y, y1, y2, y5 = G(imgs, txts)


                    pro_atrist01 = Dt(x2)
                    pro_atrist02 = Dv(y2)
                    pro_atrist11 = Dv(x5)
                    pro_atrist12 = Dt(y5)

                    G_loss = -1 / torch.mean(torch.log(1. - pro_atrist01)) + -1 / torch.mean(torch.log(1. - pro_atrist02))
                    D_loss = -torch.mean(torch.log(pro_atrist11) + torch.log(1 - pro_atrist01)) + -torch.mean(torch.log(pro_atrist12) + torch.log(1 - pro_atrist02))

                    x, y, x6, y6, l1, l2 = model(x1, y1, key1)
                    loss = calc_loss(x1, y1, x6, y6, l1, l2, labels, alpha, beta)


                    # backward + optimize only if in training phase
                    if phase == 'train':

                        G_loss.backward(retain_graph=True)
                        D_loss.backward(retain_graph=True)

                        loss.backward()
                        optimizer_G.step()
                        optimizer_Dv.step()
                        optimizer_Dt.step()
                        optimizer.step()

                        hashing_test = np.append(hashing_test, x.cpu().detach().numpy())
                        lable_test = np.append(lable_test, l2.cpu().detach().numpy())
                        test1 = hashing_test
                        test2 = lable_test
                        print(test2.shape)





                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            '''
            t_imgs, t_txts, t_labels, t_imgs_labels, t_txts_labels = [], [], [], [], []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.cuda()
                    x, y, x1, x2, x3, y1, y2, y3, x4, y4, x5, y5= model(imgs, txts, adj)
                    t_imgs.append(x3.cpu().numpy())
                    t_txts.append(y3.cpu().numpy())
                    t_imgs_labels.append(x4.cpu().numpy())
                    t_txts_labels.append(y4.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())

            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_imgs_labels = np.concatenate(t_imgs)
            t_txts_labels = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)

            img2text = calculate_top_map(t_imgs, t_txts, t_imgs_labels, t_labels)
            txt2img = calculate_top_map(t_txts, t_imgs, t_txts_labels, t_labels)
            '''


            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            # if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
            #     best_acc = (img2text + txt2img) / 2.
            #     best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
            #     test_img_acc_history.append(img2text)
            #     test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

                hashing_test = []
                lable_test = []
                hashing_test = np.array(hashing_test)
                lable_test = np.array(lable_test)
                hashing = test1.reshape(18015,64)

                #hashing_t = test1.reshape(2000, 64)
                #hashing_v = test2.reshape(2000, 64)




        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, G, Dv, Dt, hashing
