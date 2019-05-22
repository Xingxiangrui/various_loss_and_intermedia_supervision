import numpy as np
import math
import pickle
import torch
from torch import nn


class DeepMarWeights(object):
    """

    """

    def __init__(self, alpha=1.0, weight_file='./data/coco/rate.pkl'):
        self.weight_pos, self.weight_neg = self.get_weight(alpha, weight_file)

    def get_weight(self, alpha, weight_file):
        if isinstance(weight_file, str):
            with open(weight_file, 'rb') as f:
                rate = pickle.load(f)
        else:
            rate = weight_file
        weight_pos = []
        weight_neg = []
        for idx, v in enumerate(rate):
            weight_pos.append(math.exp(alpha * (1.0 - v)))
            weight_neg.append(math.exp(alpha * v))
        return weight_pos, weight_neg

    def weighted_label(self, targets_var):
        # compute the weight
        weights = torch.zeros(targets_var.shape)
        for i in range(targets_var.shape[0]):
            for j in range(targets_var.shape[1]):
                if targets_var.cpu().data[i, j] == 0:
                    weights[i, j] = self.weight_neg[j]
                elif targets_var.cpu().data[i, j] == 1:
                    weights[i, j] = self.weight_pos[j]
                else:
                    raise Exception()
        return weights


class MultiLabelSoftmax():
    def __init__(self, cond_prob):
        self.cond_prob = cond_prob

    def get_loss(self, prediction, target):
        """
        :param prediction: sigmoid output, value between 0-1, shape:[Batch, nclasses]
        :param target: value: {0,1}, shape: [Batch, nclasses]
        :return: loss
        """
        prediction = prediction.cpu().data
        target = target.cpu().data
        nrows, ncols = prediction.size()
        loss = torch.zeros(1)
        for i in range(nrows):
            nones = torch.sum(target[i, :] == 1.0)
            nzeros = ncols - nones
            neg_pred = torch.zeros(nzeros)
            weights = torch.zeros((nzeros, nones))
            l = 0
            for j in range(ncols):
                if target[i, j] == 0:
                    neg_pred[l] = prediction[i, j]
                    prob_mask = self.cond_prob[j, :] * target[i, :]
                    m = 0
                    for prob in prob_mask:
                        if prob != 0:
                            weights[l, m] = prob
                            m += 1

                    l += 1

            weights = torch.softmax(weights, dim=1)
            denominator = torch.sum(torch.exp(neg_pred.view(nzeros, 1)) * weights, dim=0)
            count = 0
            for n in range(ncols):
                if target[i, n] == 1:
                    pos_pred = torch.exp(prediction[i, n]) / (denominator[count] + torch.exp(prediction[i, n]))
                    loss = loss - torch.log(pos_pred)
                    count += 1
            loss /= torch.sum(target == 1)
        return loss


if __name__ == '__main__':
    cond_prob = torch.tensor([[0.2, 0.5, 0.4, 0.1], [0.6, 0.8, 0.2, 0.4], [0.3, 0.5, 0.2, 0.4], [0.8, 0.6, 0.1, 0.3]])
    prediction = torch.tensor([[0.8, 0.1, 0.2, 0.9], [0.1, 0.4, 0.5, 0.7], [0.1, 0.4, 0.5, 0.7]])
    target = torch.tensor([[1.0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
    demo = MultiLabelSoftmax(cond_prob)
    loss = demo.get_loss(prediction, target)
    print(loss)
