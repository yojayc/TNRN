# -*- coding:utf-8 -*-

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size], alpha is weight
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    # print("logits: ", logits.shape)
    # print("labels: ", labels.shape)

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels.float(), reduction = "none")  # BCLoss.shape is torch.Size([8, 63])

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))  # modulator.shape is torch.Size([8, 63])

    loss = modulator * BCLoss  # loss.shape is torch.Size([8, 63])

    weighted_loss = alpha * loss  # weighted_loss.shape is torch.Size([8, 63])
    focal_loss = torch.sum(weighted_loss)  # add all the losses in weighted_loss

    # 用总的样本数量归一化
    focal_loss /= torch.sum(labels)  # torch.sum(labels) is 8
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)  # weights.shape is <class 'tuple'>:(63,)
    weights = weights / np.sum(weights) * no_of_classes  # weights for each class, shape is torch.Size([63])

    # labels_one_hot.shape is torch.Size([8, 63])
    labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda()

    weights = torch.tensor(weights).float()  # torch.Size([63])
    weights = weights.unsqueeze(0).cuda()  # shape is torch.Size([1, 63])
    weights = weights.repeat(labels_one_hot.shape[0], 1)  # torch.Size([8, 63])
    weights = weights * labels_one_hot
    weights = weights.sum(1)  # 按行求和, torch.Size([8])
    weights = weights.unsqueeze(1)  # torch.Size([8, 1])
    weights = weights.repeat(1, no_of_classes)  # weights.shape is [batch_size, num_class](torch.Size([8, 63]))

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


# def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
class CB_loss1(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma):
        super(CB_loss1, self).__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.no_of_classes = no_of_classes
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        self.weights = (1.0 - beta) / np.array(effective_num)
        # print("self.weights.shape: {}\n".format(self.weights.shape))
        self.weights = self.weights / np.sum(self.weights) * self.no_of_classes
        # print("self.weights.shape: {}\n".format(self.weights.shape))
        
    def forward(self, logits, labels):
        if self.loss_type == "focal":
            labels_one_hot = F.one_hot(labels, self.no_of_classes).long().cuda()
            # print("labels_one_hot.shape: {}\n".format(labels_one_hot.shape))

            self.weights = torch.tensor(self.weights).float().cuda()
            # self.weights = torch.as_tensor(torch.from_numpy(self.weights), dtype=torch.float32).cuda()
            # self.weights = self.weights.clone().detach().cuda()

            # self.weights = self.weights.unsqueeze(0)

            # print("self.weights.shape: {}\n".format(self.weights.shape))
            # print("labels_one_hot.shape: {}\n".format(labels_one_hot.shape))
            if self.weights.shape[0] != labels_one_hot.shape[0]:
                self.weights = self.weights.repeat(labels_one_hot.shape[0], 1)
                # with open("/data/zhangbeichuan/cui/network/vit-pytorch-main/result/forward_weights.txt", 'a') as w:
                #     w.write("self.weights.shape: {}\n".format(self.weights.shape))
                #     w.write("labels_one_hot.shape: {}\n".format(labels_one_hot.shape))
                #     self.weights = self.weights.repeat(labels_one_hot.shape[0], 1)
                #     w.write("reshaped self.weights: {}\n".format(self.weights.shape))
                
            # print("self.weights.shape: {}\n".format(self.weights.shape))

            try:
                self.weights = self.weights * labels_one_hot
            except:
                print("self.weights.shape: {}\n".format(self.weights.shape))
                print("labels_one_hot.shape: {}\n".format(labels_one_hot.shape))
            
            # self.weights = self.weights
            # print("self.weights.shape: {}\n".format(self.weights.shape))
            # print("labels_one_hot.shape: {}\n".format(labels_one_hot.shape))

            # self.weights = self.weights * labels_one_hot
            self.weights = self.weights.sum(1)
            self.weights = self.weights.unsqueeze(1)
            self.weights = self.weights.repeat(1, self.no_of_classes)
        
            cb_loss = focal_loss(labels_one_hot, logits, self.weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot, weights = self.weights)
        elif self.loss_type == "softmax":
            # pred = logits.softmax(dim = 1)
            # cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = self.weights)
            # labels_one_hot = torch.LongTensor(labels_one_hot)
            # print("weights: {}\n".format(self.weights))
            # weights = torch.tensor(self.weights).float().cuda()
            weights = torch.tensor(self.weights).clone().detach().float().cuda()

            # print("logits.shape: {}\n".format(logits.shape))
            # print("labels.shape: {}\n".format(labels.shape))
            # print("weights.shape: {}\n".format(self.weights.shape))
            cb_loss = F.cross_entropy(input = logits, target = labels, weight = weights)

        return cb_loss


if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float().cuda()
    labels = torch.randint(0,no_of_classes, size = (10,)).cuda()
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "softmax"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print("cb_loss:\n", cb_loss)

    cb_loss1 = CB_loss1(samples_per_cls, no_of_classes, loss_type, beta, gamma).cuda()
    loss = cb_loss1(logits, labels)

    # print(cb_loss)
    print("cb_loss1:\n", loss)
