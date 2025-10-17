# -*- coding:utf-8 -*-
# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb
import math

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'


# 
# non-local(with half img_feature_dim, without bias, with attn_dropout) format trn
# 

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class, trn_dropout=0.8, attn_dropout=0.2):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim  # 图片的特征大小
        self.trn_dropout = trn_dropout
        self.attn_dropout = attn_dropout
            
        std = 0.01  # video-nonlocal-net/lib/core/config.py

        # remove bias to reduce overfitting
        self.g = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.theta = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.W = nn.Conv1d(in_channels=self.img_feature_dim // 2, out_channels=self.img_feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn = nn.BatchNorm1d(self.img_feature_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.attn_dropout)
        
        # initialization
        nn.init.normal_(self.g.weight, 0, std)
        nn.init.normal_(self.theta.weight, 0, std)
        nn.init.normal_(self.phi.weight, 0, std)
        
        # nn.init.constant_(self.W.weight, std)
        nn.init.normal_(self.W.weight, 0, std)
        # nn.init.constant_(self.W.bias, 0)
        
        self.classifier = self.fc_fusion()

    # 
    # x means attention weights
    # p means probability to dropout
    # 
    def attentionDropout(self, x):
        # p=0.2

        # apply dorpout according to p
        mask_weights = torch.ones(x.shape) * (1 - self.attn_dropout)
        mask_weights = torch.bernoulli(mask_weights).cuda()
        # print(mask_weights)
        # print(mask_weights.shape)

        # print(x)
        # print(mask_weights)
        x = x * mask_weights
        # print(x)

        # normalizing dropped inputs
        x_sum = torch.sum(x, dim=[1, 2])
        # print(x_sum)

        x_sum = x_sum.unsqueeze(dim=1).unsqueeze(dim=2)
        x_sum = x_sum.expand(x.shape)

        # print(x)
        x = x / x_sum
        # print(x)
        return x

    def attention_net(self, x): 
        batch_size = x.size(0)
        input_x = x
        
        # 
        # x.shape is torch.Size([batch_size, img_feature_dim, num_frames])
        # x.shape is torch.Size([4, 3, 512])
        # viewed_x.shape is torch.Size([4, 512, 3])
        # 
        # print("x: ", x.shape)
        x = x.view(batch_size, self.img_feature_dim, -1)
        # print("viewed_x: ", x.shape)
        
        theta_x = self.theta(x)  # torch.Size([4, 256, 3])
        phi_x = self.phi(x)  # torch.Size([4, 256, 3])
        g_x = self.g(x)  # torch.Size([4, 256, 3])

        # print("batch_size: ", batch_size)
        # print("theta_x: ", theta_x.shape)
        # print("phi_x: ", phi_x.shape)
        # print("g_x: ", g_x.shape)

        # d_k is img_feature_dim (256 or 512)
        d_k = theta_x.size(-2)     # d_k为query的维度
        # print("d_k: ", d_k)

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        # 
        # scaled dot-product attention
        # 
        # scores.shape is torch.Size([batch_size, num_frames, num_frames])
        # torch.Size([4, 3, 3])
        # theta_x.shape is torch.Size([4, 256, 3])
        # transposed_theta_x.shape is torch.Size([4, 3, 256])
        # phi_x.shape is torch.Size([4, 256, 3])
        # scores.shape is torch.Size([4, 3, 3])
        # 
        # print("transposed theta_x: ", theta_x.transpose(1, 2).shape)
        
        # scaled dot product
        scores = torch.matmul(theta_x.transpose(1, 2), phi_x) / d_k
        # print("scores: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维度 归一化得分
        # 
        # alpha_n is attention weight
        # alpha_n.shape is torch.Size([batch_size, num_frames, num_frames])
        #                  torch.Size([4, 3, 3])
        # F.softmax()代表归一化操作C(x)
        # 
        alpha_n = self.dropout(F.softmax(scores, dim=-1))
        # print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        # print("x: ", x.shape)
        # 
        # if self.training:
        #     alpha_n = self.attentionDropout(alpha_n)

        # 
        # alpha_n.shape is torch.Size([batch_size, num_frames, num_frames])
        # g_x.shape is torch.Size([batch_size, img_feature_dim, num_frames])
        # alpha_n.shape is torch.Size([4, 3, 3])
        # g_x.shape is torch.Size([4, 256, 3])
        # transposed g_x.shape is torch.Size([4, 3, 256])
        # context.shape is torch.Size([4, 3, 256])
        # 使用alpha_n给g_x加权
        # 
        # context.shape is torch.Size([4, 3, 256])
        context = torch.matmul(alpha_n, g_x.transpose(1, 2))
        # print("context: ", context.shape)
        # context_w.shape is torch.Size([4, 512, 3])
        # context_w = self.W(context.transpose(1, 2))
        # print("context_w: ", context_w.shape)

        # 
        # add dropout to self.W
        # 
        context = self.bn(self.W(context.transpose(1, 2)))
        # print("bn_context: ", context.shape)
        context = context.view(batch_size, context.shape[-1], context.shape[-2])
        # print("reshaped context: ", context.shape)

        context = context + input_x
        # print("context: ", context.shape)
        # context = context.sum(1)
        # print("context: ", context.shape)
        
        return context, alpha_n
        # '''

    def residual_block(self, x):
        y = x
        # print("x: ", x.shape)

        # x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.bn(x)
        # x = self.relu(x)
        x = x.view(-1, x.shape[-1], x.shape[-2])

        x, _ = self.attention_net(x)

        x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.bn(x)
        # x = self.relu(x)
        # x = x.view(-1, x.shape[-1], x.shape[-2])

        # print("x:", x.shape)

        x = x + y

        return x

    # 
    # num_frames * img_feature_dim操作将多张帧的特征融合一起输入网络层进行学习
    # 
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        # 
        # 输入网络前使用attention score融合各帧的特征
        # 而不是按相同的权重融合
        # 
        if self.trn_dropout:
            # original code with dropout       
            classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                    nn.ReLU(),
                    nn.Dropout(self.trn_dropout),
                    nn.Linear(num_bottleneck, self.num_class),
                    )
        else:
            classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                    nn.ReLU(),
                    nn.Linear(num_bottleneck, self.num_class),
                    )
        return classifier

    def forward(self, input):
        # 
        # 对输入进行reshape操作
        # input.shape is torch.Size([4, 3, 512])
        # reshaped input.shape is torch.Size([4, 1536])
        # 
        # 
        # input, attn_score = self.attention_net(input)
        # 
        # 
        # RNN-style non-local
        # 
        skip = input

        input = input.view(-1, input.shape[-1], input.shape[-2])
        # input = self.bn(input)  # 
        # input = input.view(-1, input.shape[-1], input.shape[-2])

        for i in range(self.num_frames):
            input = self.residual_block(input)
        # print("attentioned input: ", input.shape)

        input = self.bn(input)  # 
        # input = self.relu(input)  # 
        input = input.view(-1, input.shape[-1], input.shape[-2])
        input = input + skip
        
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        # print("reshaped input: {}\n".format(input.shape))
        input = self.classifier(input)
        # print("output: {}\n".format(input.shape))

        # if self.training:
            # print("in training")

        return input


class RelationModule2(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class, trn_dropout=0.8, attn_dropout=0.2):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim  # 图片的特征大小
        self.trn_dropout = trn_dropout
        self.attn_dropout = attn_dropout

        std = 0.01  # video-nonlocal-net/lib/core/config.py

        # remove bias to reduce overfitting
        self.g = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.theta = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv1d(in_channels=self.img_feature_dim, out_channels=self.img_feature_dim // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.W = nn.Conv1d(in_channels=self.img_feature_dim // 2, out_channels=self.img_feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn = nn.BatchNorm1d(self.img_feature_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.attn_dropout)

        # initialization
        nn.init.normal_(self.g.weight, 0, std)
        nn.init.normal_(self.theta.weight, 0, std)
        nn.init.normal_(self.phi.weight, 0, std)
        
        # nn.init.constant_(self.W.weight, std)
        nn.init.normal_(self.W.weight, 0, std)
        # nn.init.constant_(self.W.bias, 0)

        self.classifier = self.fc_fusion()

    # 
    # x means attention weights
    # p means probability to dropout
    # 
    def attentionDropout(self, x):
        # p=0.2

        # apply dorpout according to p
        mask_weights = torch.ones(x.shape) * (1 - self.attn_dropout)
        mask_weights = torch.bernoulli(mask_weights).cuda()
        # print(mask_weights)
        # print(mask_weights.shape)

        # print(x)
        # print(mask_weights)
        x = x * mask_weights
        # print(x)

        # normalizing dropped inputs
        x_sum = torch.sum(x, dim=[1, 2])
        # print(x_sum)

        x_sum = x_sum.unsqueeze(dim=1).unsqueeze(dim=2)
        x_sum = x_sum.expand(x.shape)

        # print(x)
        x = x / x_sum
        # print(x)
        return x

    def attention_net(self, x): 
        batch_size = x.size(0)
        input_x = x
        
        # 
        # x.shape is torch.Size([batch_size, img_feature_dim, num_frames])
        # x.shape is torch.Size([4, 3, 512])
        # viewed_x.shape is torch.Size([4, 512, 3])
        # 
        # print("x: ", x.shape)
        x = x.view(batch_size, self.img_feature_dim, -1)
        # print("viewed_x: ", x.shape)
        
        theta_x = self.theta(x)  # torch.Size([4, 256, 3])
        phi_x = self.phi(x)  # torch.Size([4, 256, 3])
        g_x = self.g(x)  # torch.Size([4, 256, 3])

        # print("batch_size: ", batch_size)
        # print("theta_x: ", theta_x.shape)
        # print("phi_x: ", phi_x.shape)
        # print("g_x: ", g_x.shape)

        # d_k is img_feature_dim (256 or 512)
        d_k = theta_x.size(-2)     # d_k为query的维度
        # print("d_k: ", d_k)

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        # 
        # scaled dot-product attention
        # 
        # scores.shape is torch.Size([batch_size, num_frames, num_frames])
        # torch.Size([4, 3, 3])
        # theta_x.shape is torch.Size([4, 256, 3])
        # transposed_theta_x.shape is torch.Size([4, 3, 256])
        # phi_x.shape is torch.Size([4, 256, 3])
        # scores.shape is torch.Size([4, 3, 3])
        # 
        # print("transposed theta_x: ", theta_x.transpose(1, 2).shape)
        
        scores = torch.matmul(theta_x.transpose(1, 2), phi_x) / math.sqrt(d_k)
        # print("scores: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维度 归一化得分
        # 
        # alpha_n is attention weight
        # alpha_n.shape is torch.Size([batch_size, num_frames, num_frames])
        #                  torch.Size([4, 3, 3])
        # 
        alpha_n = self.dropout(F.softmax(scores, dim=-1))
        # print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        # print("x: ", x.shape)
        # 
        # if self.training:
        #     alpha_n = self.attentionDropout(alpha_n)

        # 
        # alpha_n.shape is torch.Size([batch_size, num_frames, num_frames])
        # g_x.shape is torch.Size([batch_size, img_feature_dim, num_frames])
        # alpha_n.shape is torch.Size([4, 3, 3])
        # g_x.shape is torch.Size([4, 256, 3])
        # transposed g_x.shape is torch.Size([4, 3, 256])
        # context.shape is torch.Size([4, 3, 256])
        # 使用alpha_n给g_x加权
        # 
        # context.shape is torch.Size([4, 3, 256])
        context = torch.matmul(alpha_n, g_x.transpose(1, 2))
        # print("context: ", context.shape)
        # context_w.shape is torch.Size([4, 512, 3])
        # context_w = self.W(context.transpose(1, 2))
        # print("context_w: ", context_w.shape)

        # 
        # add dropout to self.W
        # 
        context = self.bn(self.W(context.transpose(1, 2)))
        # print("bn_context: ", context.shape)
        context = context.view(batch_size, context.shape[-1], context.shape[-2])
        # print("reshaped context: ", context.shape)

        context = context + input_x
        # print("context: ", context.shape)
        # context = context.sum(1)
        # print("context: ", context.shape)
        
        return context, alpha_n
        # '''

    def residual_block(self, x):
        y = x
        # print("x: ", x.shape)

        # x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(-1, x.shape[-1], x.shape[-2])

        x, _ = self.attention_net(x)

        x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.bn(x)
        x = self.relu(x)
        # x = x.view(-1, x.shape[-1], x.shape[-2])

        # print("x:", x.shape)

        x = x + y

        return x

    # 
    # num_frames * img_feature_dim操作将多张帧的特征融合一起输入网络层进行学习
    # 
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        # 
        # 输入网络前使用attention score融合各帧的特征
        # 而不是按相同的权重融合
        # 
        if self.trn_dropout:           
            classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                    nn.ReLU(),
                    nn.Dropout(self.trn_dropout),
                    nn.Linear(num_bottleneck, self.num_class),
                    )
        else:
            classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                    nn.ReLU(),
                    nn.Linear(num_bottleneck, self.num_class),
                    )
        return classifier

    def forward(self, input):
        # 
        # 对输入进行reshape操作
        # input.shape is torch.Size([4, 3, 512])
        # reshaped input.shape is torch.Size([4, 1536])
        # 
        # 
        # input, attn_score = self.attention_net(input)
        # 
        # 
        # RNN-style non-local
        # 
        skip = input

        input = input.view(-1, input.shape[-1], input.shape[-2])
        input = self.bn(input)
        # input = input.view(-1, input.shape[-1], input.shape[-2])

        for i in range(self.num_frames):
            input = self.residual_block(input)
        # print("attentioned input: ", input.shape)

        input = self.bn(input)
        input = self.relu(input)
        input = input.view(-1, input.shape[-1], input.shape[-2])
        input = input + skip
        
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        # print("reshaped input: {}\n".format(input.shape))
        input = self.classifier(input)
        # print("output: {}\n".format(input.shape))

        # if self.training:
            # print("in training")

        return input


class RelationModule1(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim  # 图片的特征大小
        self.classifier = self.fc_fusion()

    def attention_net(self, x, query, mask=None): 
        d_k = query.size(-1)     # d_k为query的维度
        # print("d_k: ", d_k)

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        # 
        # scaled dot-product attention
        # 
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        # print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1) 
        # print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        # print("x: ", x.shape)
        context = torch.matmul(alpha_n, x)
        # print("context: ", context.shape)
        # context = context.sum(1)
        # print("context: ", context.shape)
        
        return context, alpha_n

    # 
    # num_frames * img_feature_dim操作将多张帧的特征融合一起输入网络层进行学习
    # 
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        # 
        # 输入网络前使用attention score融合各帧的特征
        # 而不是按相同的权重融合
        # 
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier

    def forward(self, input):
        # 
        # 对输入进行reshape操作
        # input.shape is torch.Size([4, 3, 512])
        # reshaped input.shape is torch.Size([4, 1536])
        # 
        # print("input: {}\n".format(input.shape))
        input, attn_score = self.attention_net(input, input)
        # print("attentioned input: ", input.shape)

        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        # print("reshaped input: {}\n".format(input.shape))
        input = self.classifier(input)
        # print("output: {}\n".format(input.shape))

        return input


class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim  # 512
        # 倒着输出帧的数量
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        # 
        # 返回不同scale下的帧序
        # 
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist

        for i in range(len(self.scales)):
            scale = self.scales[i]  # scale表示帧数
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            # 用于融合多个帧范围的特征
            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    # input.shape is torch.Size([batch_size, num_frames, img_feature_dim])
    def forward(self, input):
        # 
        # the first one is the largest scale
        # 第一个relations_scales是最长帧序列
        # 
        # [:, self.relations_scales[0][0]]选择第一个relations中对应帧编号的特征
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            # numpy.random.choice()从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回
            # np.random.choice(a, size=None, replace=True, p=None)
            # 从a中以概率p随机选择size个，p没有指定时相当于是一致的分布
            # replacement代表抽样之后还放不放回去，如果是False，则抽出的数都不一样
            # 如果是True，有可能会出现重复的
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                # 
                # 直接将多个序列长度的特征相加
                # 
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        # itertools库主要用于排列组合
        import itertools
        # itertools.combinations求列表或生成器中指定数目的元素不重复的所有组合
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class RelationModuleMultiScaleWithClassifier(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


def return_TRN(relation_type, img_feature_dim, num_frames, num_class, trn_dropout=None, attn_dropout=None):
    if relation_type == 'TRN':
        TRNmodel = RelationModule(img_feature_dim, num_frames, num_class, trn_dropout, attn_dropout)
    elif relation_type == 'TRNmultiscale':
        TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    else:
        raise ValueError('Unknown TRN' + relation_type)

    return TRNmodel


if __name__ == "__main__":
    batch_size = 4
    num_frames = 3
    num_class = 63
    img_feature_dim = 512

    input_var = torch.rand(batch_size, num_frames, img_feature_dim).cuda()
    print("input_var: {}\n".format(input_var.shape))
    # model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)

    model = RelationModule(img_feature_dim, num_frames, num_class).cuda()
    # model.eval()
    # model.dropout()

    output = model(input_var)
    print("output: {}\n".format(output.shape))
