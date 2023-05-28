import torch
from torch import nn
from torch.nn import Parameter, init
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def linjie(l1, l2):

    #l2 = l1.t()

    adj = torch.matmul(l1, l2)
    one = torch.ones_like(adj)
    zero = torch.zeros_like(adj)
    return torch.where(adj > 0, one, zero)

    return adj


def normalize1(A, symmetric=True):
    # A = A+I
    A = torch.as_tensor(A, device='cpu')
    #A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return ((D.T).mm(A).T).mm(D.T)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

def normalize(A, symmetric=True):
        # A = A+I
        A = torch.as_tensor(A, device='cpu')
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


class Dt(nn.Module):
    def __init__(self, input=256):
        super(Dt, self).__init__()
        self.f1 = nn.Linear(input, 128)
        self.ReLU = nn.ReLU()
        self.f2 = nn.Linear(128, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.ReLU(x)
        x = self.f2(x)
        x = self.output(x)
        return x

class Dv(nn.Module):
    def __init__(self, input=256):
        super(Dv, self).__init__()
        self.f1 = nn.Linear(input, 128)
        self.ReLU = nn.ReLU()
        self.f2 = nn.Linear(128, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.ReLU(x)
        x = self.f2(x)
        x = self.output(x)
        return x




class G(nn.Module):
    """Network to learn image representations"""

    def __init__(self, inputv=4096, inputt=1386):
        super(G, self).__init__()
        self.img_net = ImgNN(256)
        self.text_net = TextNN(inputt, 256)
        self.f11 = nn.Linear(256, 128)
        self.ReLU = nn.ReLU()
        self.f12 = nn.Linear(256, 128)

        self.f41 = nn.Linear(128, 256)
        self.f42 = nn.Linear(128, 256)


    def forward(self, x, y):
        x = self.img_net(x)
        x5 = x
        x = self.ReLU(self.f11(x))

        x1 = x

        x = self.ReLU(self.f41(x))
        x2 = x

        y = self.text_net(y)
        y5 = y
        y = self.ReLU(self.f12(y))

        y1 = y

        y = self.ReLU(self.f42(y))
        y2 = y

        return x, x1, x2, x5, y, y1, y2, y5


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, adj):
        weight= self.weight.double()
        support = torch.matmul(input1, weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=256):
        super(ImgNN, self).__init__()
        #self.denseL1 = nn.Linear(input_dim, output_dim)

        self.alexnet = models.alexnet(pretrained=True)

        self.fc_encode = nn.Linear(1000, output_dim)
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)

        out = F.relu(self.alexnet(x))


        out = F.relu(self.fc_encode(out))



        return out
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1386, output_dim=512):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)

        out = F.relu(self.denseL1(x))

        return out
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)


class FLM(nn.Module):
    def __init__(self, inputl = 24, inputg = 256, outputg = 64, dropout = 0):
        super(FLM, self).__init__()
        '''
        self.input1 = models.resnet50(pretrained=True)
        self.input1.fc = nn.Linear(2048, 4096)
        '''
        '''
        self.img_net = ImgNN(inputv, 512)
        self.text_net = TextNN(inputt, 512)
        self.f11 = nn.Linear(512, 256)
        self.ReLU = nn.ReLU()
        self.f12 = nn.Linear(512, 256)

        self.f41 = nn.Linear(256, 512)
        self.f42 = nn.Linear(256, 512)
        '''
        self.f51 = nn.Linear(128, 128)
        self.f52 = nn.Linear(128, 128)
        self.fc = nn.Linear(inputg, inputl)
        self.fc1 = nn.Linear(outputg, inputl)
        self.sig = nn.Sigmoid()
        self.gc1 = GraphConvolution(inputg, 128)
        self.gc2 = GraphConvolution(128, 128)
        self.gc3 = GraphConvolution(128, outputg)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = dropout


    def forward(self, x, y, k, n=0):


        x6 = self.relu(self.f51(x))
        y6 = self.relu(self.f52(y))



        if n == 0:
            x6 = torch.mul(x6, torch.mul(y6, x6))
            y6 = torch.mul(y6, torch.mul(x6, y6))
            f = torch.cat((x6, y6), 1)
        elif n == 1:
            x6 = torch.mul(x6, torch.mul(y6, x6))
            y6 = torch.mul(y6, torch.mul(x6, y6))
            f = torch.cat((x6, y6), 1)
        else:
            x6 = torch.mul(x6, torch.mul(y6, x6))
            y6 = torch.mul(y6, torch.mul(x6, y6))
            f = torch.cat((y6, y6), 1)


        l1 = self.sig(self.fc(f))


        #x = self.tanh(self.f51(x1))

        #y = self.tanh(self.f51(y1))
        #'''
        k = torch.as_tensor(k, dtype=float, device='cuda:0')
        l1 = torch.as_tensor(l1, dtype=float, device='cuda:0')
        f = torch.as_tensor(f, dtype=float, device='cuda:0')

        adj1 = linjie(l1, k.T)
        adj2 = linjie(k, k.T)
        adj2 = normalize(adj2)
        adj3 = linjie(k, l1.T)




        adj1 = torch.as_tensor(adj1, device='cuda:0')
        adj2 = torch.as_tensor(adj2, device='cuda:0')
        adj3 = torch.as_tensor(adj3, device='cuda:0')

        adj1 = adj1.double()


        x = F.relu(self.gc1(f, adj1.T))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.tanh(self.gc3(x, adj3.T))
        #'''
        x = x.float()

        l2 = self.sig(self.fc1(x))

        return x, y, x6, y6, l1, l2


def show_Hyperparameter(args):
    argsDict = args.__dict__
    print(argsDict)
    print('the settings are as following:')
    for key in argsDict:
        print(key, ':', argsDict[key])