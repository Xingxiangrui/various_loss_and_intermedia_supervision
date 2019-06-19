# various_loss_and_intermedia_supervision
various loss and intermedia supervision and supplement loss

add model with intermedia loss

initial model has only one loss

we add a loss from the intermedia of the network.

背景：网络最终的预测结果作为loss，可以继续添加中间loss做为

思路：增加原始网络中继输出——中继输出与标签之间运算loss——与原始loss想加做为最终loss

博主代码地址：https://github.com/Xingxiangrui/various_loss_and_intermedia_supervision

目录

一、原始loss的运算

1.1 loss位置

1.2 criterion定义

1.3 optimizer定义

二、GAT更改

2.1 原始GAT

2.2 去掉残差

三、中继输出

3.1 原始网络结构

3.2 定义新的supplement layer

3.3 网络结构中添加

四、结构更改

4.1 嵌套顺序

4.2 trainer中

4.3 loss
一、原始loss的运算
1.1 loss位置

    criterion = util.get_criterion(args)
    model = util.get_model(args)
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.LR, args.LRP),
                                momentum=args.MOMENTUM,
                                weight_decay=args.WEIGHT_DECAY)

[点击并拖拽以移动]

通过 get_criterion定义loss

optimizer为随机提督下降算法，送入train

    trainer = T.Trainer(args, train_dataloader, val_dataloader, optimizer, model, criterion, lr_scheduler)
    trainer.run()

其中包括了 optimizer，criterion
1.2 criterion定义

即相应的loss定义：

def get_criterion(args):
    if args.LOSS_TYPE == 'MultiLabelSoftMarginLoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.LOSS_TYPE == 'DeepMarLoss':
        criterion = F.binary_cross_entropy_with_logits
    else:
        raise Exception()
    return criterion

我们常用的deepMarLoss,

区别是：softmax_cross_entropy_with_logits 要求传入的 labels 是经过 one_hot encoding 的数据，而 sparse_softmax_cross_entropy_with_logits 不需要。

https://www.jianshu.com/p/47172eb86b39

源码：

@weak_script
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str, Optional[Tensor]) -> Tensor
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)

1.3 optimizer定义

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.LR, args.LRP),
                                momentum=args.MOMENTUM,
                                weight_decay=args.WEIGHT_DECAY)

二、GAT更改
2.1 原始GAT

激活之前，加了一个h_prime与 self.beta*h，相当于一个残差结构。

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

2.2 去掉残差

直接将相加的部分删掉即可

        # out = F.elu(h_prime + self.beta * h) # residual format
        out = F.elu(h_prime)  # without residual, only elu

三、中继输出
3.1 原始网络结构

原始的网络结构需要分支出一个辅助输出。

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

        # GAT between classes
        # [ batch, n_classes*class_channels= 80*256 = 20480 ]
        x = self.gat(x)

        # output  [ batch , n_classes ]
        x= self.final_fcs(x).view(x.size(0),x.size(1))

        return x

3.2 定义新的supplement layer

我们需要从送入GAT之前的feature进行预测与输出。

输入：[ batch, n_classes, class_channels ]

输出：[batch, n_classes]

        # fixme  torch parallel final fcs
        self.final_fcs=nn.Sequential(
                parallel_final_fcs_ResidualLinearBlock(n_classes=nclasses, reduction=2, class_channels=class_channels),
                parallel_output_linear(n_classes=nclasses,  class_channels=class_channels) )

        # fixme  supplement output structure
        self.supplement_output_structure=nn.Sequential(
            parallel_final_fcs_ResidualLinearBlock(n_classes=nclasses, reduction=2, class_channels=class_channels),
            parallel_final_fcs_ResidualLinearBlock(n_classes=nclasses, reduction=2, class_channels=class_channels),
            parallel_output_linear(n_classes=nclasses, class_channels=class_channels)
        )

3.3 网络结构中添加

多一个输出，supplement_out做为网络预测的输出。

        # group linear from group to classes
        # [ Batch, n_groups=12, group_channels=512 ] ->  [ batch, n_classes, class_channels ]
        x=self.group_linear(x)

        # supplement structure for supplement loss
        # input size same as input of gat :[ batch, n_classes, class_channels ]
        # output size same as model output : [ batch , n_classes ]
        supplement_out=self.supplement_output_structure(x)

        # GAT between classes
        # input,output: [ batch, n_classes, class_channels ]
        x = self.gat(x)

        # output  [ batch , n_classes ]
        x= self.final_fcs(x).view(x.size(0),x.size(1))

        return x,supplement_out

四、结构更改
4.1 嵌套顺序

train.py中，将模型改为：

    MODEL = 'group_clsgat_with_supple_loss'

在util.py之中，添加

    elif args.MODEL == 'group_clsgat_with_supple_loss':
        import models.group_clsgat_with_supple_loss as group_clsgat_parallel
        model = group_clsgat_parallel.GroupClsGat(args.BACKBONE, groups=args.GROUPS, nclasses=args.NCLASSES,
                                                  nclasses_per_group=args.NCLASSES_PER_GROUP,
                                                  group_channels=args.GROUP_CHANNELS,
                                                  class_channels=args.CLASS_CHANNELS)

4.2 trainer中

更改模型运行的结果，如果需要增加结构，则模型两个预测输出。

            # compute output
            if(self.arch=='group_clsgat_with_supple_loss'):
                output,supplement_out = model(image, embedding)
            else:
                output = model(image, embedding)

增加中间输出
4.3 loss

增加将结果的loss增加运算。

            # fixme compute new loss
            if (self.arch == 'group_clsgat_with_supple_loss'):
                if self.loss_type == 'DeepMarLoss':
                    weights = self.deepmar_loss.weighted_label(target.detach())
                    if torch.cuda.is_available():
                        weights = weights.cuda()
                    output_loss = criterion(output, target, weight=weights)
                    supplement_loss=criterion(supplement_out, target, weight=weights)
                    loss=output_loss+0.1*supplement_loss
                else:
                    output_loss=criterion(output, target)
                    supplement_loss=criterion(supplement_out, target)
                    loss = output_loss+0.1*supplement_loss
            else:
                if self.loss_type == 'DeepMarLoss':
                    weights = self.deepmar_loss.weighted_label(target.detach())
                    if torch.cuda.is_available():
                        weights = weights.cuda()
                    loss = criterion(output, target, weight=weights)
                else:
                    loss = criterion(output, target)




