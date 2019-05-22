"""
created by Xingxiangrui on 2019.5.22
    parallelism of group class gat model
    hierarchical graph attention network, grouping network use conv.
    add model structure with supplement loss
"""

import torchvision.models as models
import models.utils as utils
import torch
from torch import nn
import torch.nn.functional as F
import math


class BGATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        """
        :param in_features: input's features
        :param out_features: output's features
        :param dropout: attention dropout.
        :param alpha:
        :param is_activate:
        """
        super(BGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform(self.W.data, gain=1.414)  # fixme
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform(self.a.data, gain=1.414)  # fixme
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.beta = nn.Parameter(data=torch.ones(1))
        self.beta = nn.Parameter(data=torch.ones(1))  # fixme!!!!!!

        # self.register_parameter('beta', self.beta)

    def forward(self, x):
        # [B,N,C]
        B, N, C = x.size()
        # h = torch.bmm(x, self.W.expand(B, self.in_features, self.out_features))  # [B,N,C]
        h = torch.matmul(x, self.W)  # [B,N,C]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                                  2 * self.out_features)  # [B,N,N,2C]
        # temp = self.a.expand(B, self.out_features * 2, 1)
        # temp2 = torch.matmul(a_input, self.a)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N]

        attention = F.softmax(attention, dim=2)  # [B,N,N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C]-> [B,N,C]
        # out = F.elu(h_prime + self.beta * h) # residual format
        out = F.elu(h_prime)  # without residual, only elu
        return out


class NHeadsBGATLayer(nn.Module):
    def __init__(self, nheads, aggregate, in_features, out_features, dropout=0, alpha=0.2):
        super(NHeadsBGATLayer, self).__init__()
        self.aggregate = aggregate
        self.gats = nn.ModuleList(
            [BGATLayer(in_features=in_features, out_features=out_features, dropout=dropout, alpha=alpha) for _ in
             range(nheads)])

    def forward(self, x):
        if self.aggregate == 'mean':
            x = torch.stack([att(x) for att in self.attentions], dim=2)
            x = torch.mean(x, dim=3).squeeze(3)
        elif self.aggregate == 'concat':
            x = torch.cat([att(x) for att in self.attentions], dim=2)  # [B,N,nheads*C]
        else:
            raise Exception()
        return x


# fixme , torch parallel bottlenect structure
class Parallel_Bottleneck(nn.Module):
    expansion = 4

    # groups=12 group_channels=512
    def __init__(self, groups, group_channels, stride=1, downsample=None):
        super(Parallel_Bottleneck, self).__init__()
        expand = 2  # fixme
        self.groups=groups
        # squeeze from group_channels to group_channels//2*groups
        self.conv1 = nn.Conv2d(group_channels, groups*group_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(groups*group_channels)
        self.conv2 = nn.Conv2d(groups*group_channels, groups*group_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=12)
        self.bn2 = nn.BatchNorm2d(groups*group_channels)
        self.conv3 = nn.Conv2d(groups*group_channels, groups*group_channels, kernel_size=1, bias=False, groups=12)  # fixme
        self.bn3 = nn.BatchNorm2d(groups*group_channels)  # fixme
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(planes * expand)  # fixme
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # input x=residual[batch, group_channels=512, W,H]
        # residual = torch.Tensor(x.size(0),0,x.size(2),x.size(3))
        # residual [batch, groups*group_channels=6144, W,H]
        for group_idx in range(self.groups):
            if (group_idx==0):
                residual=x
            else:
                residual=torch.cat((residual,x),dim=1)

        # squeeze and to gropus=12 [batch, group_channels//2*groups=512//2*12=3072,W,H]
        out = self.conv1(x)
        # same as above [batch, group_channels//2*groups=512//2*12=3072,W,H]
        out = self.bn1(out)
        # same as above [batch, group_channels//2*groups=512//2*12=3072,W,H]
        out = self.relu(out)

        # same as above [batch, group_channels//2*groups=512//2*12=3072,W,H]
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # expand [batch , groups_channels*groups=6144, W,H ]
        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        # residual [batch, groups*group_channels=6144, W,H]
        out += residual
        out = self.relu(out)

        return out

# fixme   torch parallen group linear, from groups fc to classes
class Parallel_GroupLinear(nn.Module):
    def __init__(self, n_groups, n_classes, group_channels,class_channels,nclasses_per_group, bias=True):
        super(Parallel_GroupLinear, self).__init__()
        self.n_groups = n_groups
        self.n_classes=n_classes
        self.group_channels = group_channels
        self.class_channels = class_channels
        self.nclasses_per_group=nclasses_per_group
        self.weight = nn.Parameter(torch.Tensor(group_channels,n_classes,class_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes,class_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input [batch, groups=12, group_channels ]
        # groups_input=torch.Tensor(input.size(0),0,self.group_channels)

        for group_idx in range(len(self.nclasses_per_group)):
            for class_per_group_idx in range(self.nclasses_per_group[group_idx]):
                if ((group_idx==0)and(class_per_group_idx==0)):
                    groups_input=input[:,group_idx,:].view(input.size(0),1,self.group_channels)
                else:
                    groups_input=torch.cat((groups_input,input[:,group_idx,:].view(input.size(0),1,self.group_channels)),dim=1)
        # groups input [ batch, n_classes, group_channels ]

        # [ batch, n_classes, group_channels-> class_channels ]
        # b:batch, n:n_classes, c:group_channels, k:class_channels
        output = torch.einsum('bnc,cnk->bnk', [groups_input, self.weight])

        if self.bias is not None:
            output = output + self.bias

        # [ batch, n_classes, class_channels ]
        return output

    def extra_repr(self):
        return 'ngroups={},in_features={}, out_features={}, bias={}'.format(self.ngroups,
                                                                            self.in_features, self.out_features,
                                                                            self.bias is not None
                                                                            )
# fixme  group divided fc
class group_divided_linear(nn.Module):
    def __init__(self, ngroups, in_features, out_features, bias=True):
        super(group_divided_linear, self).__init__()
        self.ngroups = ngroups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(ngroups, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ngroups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # b:batch, g:groups, i:in_features, o:out_features
        output = torch.einsum('bgi,gio->bgo', [input, self.weight])
        if self.bias is not None:
            output = output + self.bias

        return output


# fixme  final parallel fcs residual Linear block
class nclass_BasicLinear(nn.Module):
    # in_features: class_channels ,  out_features: out_class_channels,  fc_groups:classes
    def __init__(self, in_features, out_features, fc_groups):
        super(nclass_BasicLinear, self).__init__()
        self.fc = group_divided_linear(in_features=in_features, out_features=out_features, ngroups=fc_groups)
        self.bn = nn.BatchNorm1d(fc_groups) #fixme maybe wrong
        self.relu = nn.ReLU()

    def forward(self, x):
        # input  [batch, fc_groups, in_features]
        # output [batch, fc_groups, out_features]
        x = self.fc(x)

        x = self.bn(x)
        x = self.relu(x)
        return x

# fixme  final fcs residual Linear block
class parallel_final_fcs_ResidualLinearBlock(nn.Module):
    def __init__(self, n_classes, class_channels):
        super(parallel_final_fcs_ResidualLinearBlock, self).__init__()
        self.fc1 = nclass_BasicLinear(in_features=class_channels, out_features=class_channels, fc_groups=n_classes)
        self.fc2 = nclass_BasicLinear(in_features=class_channels, out_features=class_channels, fc_groups=n_classes)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        out = residual + x
        return out

# fixme  final fcs without residual Linear block
class parallel_final_fcs(nn.Module):
    def __init__(self, n_classes, class_channels):
        super(parallel_final_fcs, self).__init__()
        self.fc1 = nclass_BasicLinear(in_features=class_channels, out_features=class_channels, fc_groups=n_classes)
        self.fc2 = nclass_BasicLinear(in_features=class_channels, out_features=class_channels, fc_groups=n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# fixme   fianal parallel output linear
class parallel_output_linear(nn.Module):
    def __init__(self, n_classes,  class_channels):
        super(parallel_output_linear, self).__init__()
        self.output_fc=group_divided_linear(in_features=class_channels,out_features=1, ngroups=n_classes)
    def forward(self,x):
        x=self.output_fc(x)
        return x



# fixme  torch parallel Head functions
class Head(nn.Module):
    def __init__(self, groups, nclasses, nclasses_per_group, group_channels, class_channels):
        super(Head, self).__init__()
        self.groups = groups
        self.group_channels=group_channels
        self.nclasses = nclasses
        self.nclasses_per_group = nclasses_per_group
        # input [Batch, 2048, W=14,H=14]
        # to [ Batch, group_channels=512, W=14,H=14  ]
        self.reduce_conv = utils.BasicConv(in_planes=2048, out_planes=group_channels, kernel_size=1)

        # fixme  torch parallel bottlenect
        self.bottle_nect=Parallel_Bottleneck(groups=groups,group_channels=group_channels)

        # fixme  torch parallel group fc
        self.group_linear=Parallel_GroupLinear(n_groups=groups, n_classes=nclasses, group_channels=group_channels,
                                               class_channels=class_channels,nclasses_per_group=nclasses_per_group)

        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.gat = BGATLayer(in_features=class_channels, out_features=class_channels, dropout=0, alpha=0.2)

        # fixme  torch parallel final fcs
        self.final_fcs=nn.Sequential(
                parallel_final_fcs_ResidualLinearBlock(n_classes=nclasses, class_channels=class_channels),
                parallel_output_linear(n_classes=nclasses,  class_channels=class_channels) )

        # fixme  supplement output structure
        self.supplement_output_structure=nn.Sequential(
            parallel_final_fcs(n_classes=nclasses, class_channels=class_channels),
            parallel_final_fcs(n_classes=nclasses,  class_channels=class_channels),
            parallel_output_linear(n_classes=nclasses, class_channels=class_channels)
        )

    # fixme  head forward
    def forward(self, x):
        # input x [batch_size, 2048, W=14, H=14]
        # conv from 2048 to Group_channel=512
        x=self.reduce_conv(x)
        # output x [B, group_channels=512, W=14, H=14]

        # output x [B, group_channels*groups=512*12=6144, W=14, H=14]
        x=self.bottle_nect(x)

        # output x [ Batch, n_groups=12, group_channels=512 ]
        x = self.gmp(x).view(x.size(0), self.groups,self.group_channels)

        # group linear from group to classes
        # [ Batch, n_groups=12, group_channels=512 ] ->  [ batch, n_classes, class_channels ]
        x=self.group_linear(x)

        # supplement structure for supplement loss
        # input size same as input of gat :[ batch, n_classes, class_channels ]
        # output size same as model output : [ batch , n_classes ]
        supplement_out=self.supplement_output_structure(x).view(x.size(0),x.size(1))

        # GAT between classes
        # input,output: [ batch, n_classes, class_channels ]
        x = self.gat(x)

        # output  [ batch , n_classes ]
        x= self.final_fcs(x).view(x.size(0),x.size(1))

        return x,supplement_out


# fixme  total network structure
class GroupClsGat(nn.Module):
    def __init__(self, backbone, groups, nclasses, nclasses_per_group, group_channels, class_channels):
        super(GroupClsGat, self).__init__()
        self.groups = groups
        self.nclasses = nclasses
        self.nclasses_per_group = nclasses_per_group
        self.group_channels = group_channels
        self.class_channels = class_channels

        if backbone == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif backbone == 'resnet101_cbam':
            import mymodels.cbam as cbam
            model = cbam.resnet101_cbam(pretrained=True)
        else:
            raise Exception()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4, )
        self.heads = Head(self.groups, self.nclasses, self.nclasses_per_group, group_channels=self.group_channels,
                          class_channels=self.class_channels)

    def forward(self, x, inp):
        x = self.features(x)  # [B,2048,H,W] [2,2048,14,14]
        x,supplement_out = self.heads(x)
        return x,supplement_out

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lrp},
            {'params': self.heads.parameters(), 'lr': lr},
        ]


if __name__ == '__main__':
    model = GroupClsGat(backbone='resnet101', groups=12, nclasses=80,
                        nclasses_per_group=[1, 8, 5, 10, 5, 10, 7, 10, 6, 6, 5, 7], group_channels=512,
                        class_channels=256)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')
    x = torch.zeros(2, 3, 448, 448).random_(0, 10)
    out,supplement_out = model(x, 1)
    print('model debug done...')
    # model=models.(pretrained=False)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))
