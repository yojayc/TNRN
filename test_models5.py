# -*- coding: UTF-8 -*-

import argparse
import time
import os

import numpy as np
import torch.nn.parallel
import torch.optim
import warnings
# warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet1
from dataset_tangent import TangentDataSet
# from original_models import TSN
from models_tangent_with_trn5 import TSN
from transforms import *
# from ops import ConsensusModule

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset',
                    type=str,
                    choices=['ucf101', 'hmdb51', 'kinetics', 'egok360'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
# parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_faces', type=int, default=20)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)  # choices=[1, 10]
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='TRN',
                    choices=['avg', 'TRN','TRNmultiscale'])
parser.add_argument('--tangent_crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--trn_dropout', type=float, default=0.5)
parser.add_argument('--attn_dropout', type=float, default=0.5)
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--softmax', type=int, default=0)

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'egok360':
    num_class = 63
else:
    raise ValueError('Unknown dataset ' + args.dataset)

'''
    def __init__(self, num_class, num_segments, num_faces, 
                 hidden_size, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', tangent_consensus_type='mha',
                 before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
'''
'''
print("dataset: {}".format(args.dataset))
print("num_class: {}".format(num_class))
print("modality: {}".format(args.modality))
print("test_list: {}".format(args.test_list))
print("tangent_test_list: {}".format(args.tangent_test_list))
print("weights: {}".format(args.weights))
print("args.arch: {}".format(args.arch))
'''

net = TSN(num_class, args.test_segments, 
            args.test_faces, args.modality, 
            base_model=args.arch, consensus_type=args.crop_fusion_type,
            tangent_consensus_type=args.tangent_crop_fusion_type,
            dropout=args.dropout, trn_dropout=args.trn_dropout, attn_dropout=args.attn_dropout)

checkpoint = torch.load(args.weights)
# 输入args.weights就是.pth文件
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'],
                                              checkpoint['best_prec1']))
'''
Python字典items()方法以列表返回可遍历的(键, 值)元组数组
Python join()方法用于将序列中的元素以指定的字符连接生成一个新的字符串
如下为将k中的元素去掉'.'符号后，将第二位到最后一位重新以'.'连接赋给base_dict
'''
base_dict = {
    '.'.join(k.split('.')[1:]): v
    for k, v in list(checkpoint['state_dict'].items())
}
# print("base_dict:{}".format(base_dict))
'''
load_state_dict(state_dict)
将state_dict中的parameters和buffers复制到此module和它的后代中
state_dict中的key必须和  model.state_dict()返回的key一致
NOTE：用来加载模型参数

parameters:
    state_dict (dict) – 保存parameters和persistent buffers的字典
'''
net.load_state_dict(base_dict)

if args.test_crops == 1:
    # 先resize到指定尺寸,然后再做center crop操作，最后得到的是net.input_size的尺寸,图片做完crop一系列操作后输出还是一张照片
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
    # print("cropping(GroupScale,GroupCenterCrop):{}".format(cropping))
elif args.test_crops == 10:
    # 调用该项目下的transforms.py脚本中的GroupOverSample类进行重复采样的crop操作，输出五张crop图像，五张crop加左右翻转
    cropping = torchvision.transforms.Compose(
        [GroupOverSample(net.input_size, net.scale_size)])
    # print("cropping(GroupOverSample):{}".format(cropping))
else:
    raise ValueError(
        "Only 1 and 10 crops are supported while we got {}".format(
            args.test_crops))

# print("args.test_list:{}".format(args.test_list))

# data_loader is a DataLoader object
data_loader = torch.utils.data.DataLoader(
    TangentDataSet(args.test_list,
                   num_segments = args.test_segments,
                   num_faces = args.test_faces,
                   random_shift=False,
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
    batch_size=1,
    shuffle=False,
    num_workers=args.workers * 2,
    pin_memory=True)

# print("data_loader:{}".format(data_loader))

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

# cuda(device_id=None)
# 将所有的模型参数(parameters)和buffers赋值GPU
# net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()
net.eval()  # 将模型设置成evaluation模式，仅仅当模型中有Dropout和BatchNorm时才会有影响

data_gen = enumerate(data_loader)
# print('The data_gen are :',data_gen)

total_num = len(data_loader.dataset)
print('The total_num is :', total_num)
# print('dataset: ',data_loader.dataset)
c = 0
output = []


def eval_video(video_data):
    # video frame path, video frame number, and video groundtruth class.
    # 
    # in train and eval mode
    # 
    # img_input.shape is torch.Size([6, 9, 224, 224])
    # tangent_input.shape is torch.Size([6, 135, 224, 224])
    # 
    # in test mode
    # 
    # img_data.shape is torch.Size([1, 750, 224, 224])
    # tangent_img_data.shape is torch.Size([1, 11250, 224, 224]), 750*15=11250
    # 
    i, img_data, label = video_data

    #
    # img_data.shape is torch.Size([1, 300, 224, 224])
    #

    # print("img_data: {}".format(img_data.shape))
    # print("tangent_data: {}".format(tangent_data.shape))

    num_crop = args.test_crops

    # print("video data:{}".format(video_data))
    # print("###################################################################")
    # print("i:{},data:{},label:{},num_crop:{}".format(i, data, label, num_crop))

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality " + args.modality)

    # 
    # batch_size is 250
    # reshaped img_data.shape is torch.Size([250, 3, 224, 224])
    # reshaped tangent_data.shape is torch.Size([250, 45, 224, 224])
    # 
    # reshaped img_data.shape: torch.Size([100, 3, 224, 224])
    # 
    img_data = img_data.view(-1, length, img_data.size(2), img_data.size(3))

    # print("reshaped img_data: {}".format(img_data.shape))
    # print("reshaped tangent_data: {}".format(tangent_data.shape))
    '''
    args.test_crops = 10
    args.test_segments = 25
    channel = 3
    将原本的输入(1,3*args.test_crops*args.test_segments,224,224)变换到
    (args.test_crops*args.test_segments,3,224,224)，相当于batch size为
    args.test_crops*args.test_segments
    '''
    rst = net(img_data).data.cpu().numpy().copy()
    # print("rst:{}".format(rst.shape))
    '''net(input_var)得到的结果是Variable，如果要读取Tensor内容，
    需读取data变量，cpu()表示存储到cpu，numpy()表示Tensor转为
    numpy array，copy()表示拷贝
    '''
    # print("num_crop:{}, args.test_segments:{}, num_class:{}".format(
    #       num_crop, args.test_segments, num_class))

    # num_crop:1, args.test_segments:25, num_class:63
    # rst1.shape is torch.Size([1, 25, 63])
    # rst2.shape is torch.Size([25, 63])
    # rst3.shape is torch.Size([25, 1, 63])
    # 
    rst1 = rst.reshape((num_crop, 1, num_class))
    # rst1 = rst.reshape((num_crop, num_class))
    rst2 = rst1.mean(axis=0)
    rst3 = rst2.reshape((1, 1, num_class))
    
    # print("rst1: {}".format(rst1.shape))
    # print("rst2: {}".format(rst2.shape))
    # print("rst3: {}".format(rst3.shape))
    
    return i, rst3, label[0]


'''rst.reshape((num_crop, args.test_segments, num_class))表示将输入维数（二维）变化到
指定维数（三维），mean(axis=0)表示对num_crop维度取均值，也就是原来对某帧图像的10张crop
或clip图像做预测，最后是取这10张预测结果的均值作为该帧图像的结果
最后返回的是3个值，分别表示video的index，预测结果和video的真实标签。
'''
proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (img_data, label) in data_gen:
    # print("i:{}, (data, label):{}".format(i, (data, label)))
    # print("data:{}\nlabel:{}".format(data,label))
    if i >= max_num:
        break
    rst = eval_video((i, img_data, label))
    # print("rst[1:]: {}".format(rst[1:].shape))
    # rst[0] shape:(25, 1, 101)
    output.append(rst[1:])  # exclude i
    # print("output:{}".format(output))
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(
        i, i + 1, total_num,
        float(cnt_time) / (i + 1)))
'''
c = 0
print("output length:{}".format(len(output)))
for x in output:
    print("count:{}, x:{}".format(c, x))
    print("x[0]'s size':{}".format(x[0].size))
    print("x[0]'s shape:{}".format(x[0].shape))
    print("x[0]:{}".format(x[0]))
    print("x[1]:{}".format(x[1]))
    m = np.mean(x[0], axis=0)
    print("mean(x[0])'s shape:{}".format(m.shape))
    print("mean(x[0])'s length:{}".format(m.size))
    maxNum = m.max()
    maxPos = np.argmax(m)
    print("np.mean(x[0], axis=0):{}".format(m))
    print("max num is {}".format(maxNum))
    print("max num locate in {}".format(maxPos))

    c = c + 1
'''

# numpy.argmax(a, axis=None, out=None), starts from zero
# Returns the indices of the maximum values along an axis.
# 返回最大数的索引
# axis = 0: 压缩行，对各列求均值
# video_pred记录预测的视频分类结果
#
# x[0]'s shape(25, 1, 101)
# np.mean(x[0], axis=0)'s shape is (1, 101)
# video_pred为预测类别
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
# print("video_pred:{}".format(video_pred))

# x[0]为预测得分
# x[1]为实际类别
video_labels = [x[1] for x in output]

length = np.arange(total_num)
# index记录预测标记与实际标记不符的数据位置
index = length[np.array(video_labels) != np.array(video_pred)]
data_labels = [video_labels[i] for i in index]
data_pred = [video_pred[i] for i in index]

# 找出预测类别与实际类别不相符的视频片段
# 并记录文件路径
lines = []
c = 0
with open(args.test_list, 'r') as x:
    line = x.readlines()
    for i in index:
        line[i] = line[i].strip('\n')
        lines.insert(c, line[i] + " " + str(video_pred[i]))
        c += 1
    s = '\n'.join(lines)

str = args.test_list.split('/')
result = "result_" + str[-1]

with open(result, 'w') as z:
    z.write(s)

# print("s:{}".format(s))

# print("data_labels:{}\ndata_pred:{}".format(data_labels,data_pred))
# print("video_labels:{}".format(video_labels))
# print('----'*30)
# print('The length of video_labels:',len(video_labels))
# print('The length of video_pred:',len(video_pred))
# print('The video_labels are :',video_labels)
# print('The video_pred are :', video_pred)
# print("right or wrong")
#num = np.arange(0,37)  # 共有38个测试视频类别
# 打印出预测标记与真实标记不符的测试视频序号
#print(num[np.array(video_labels)!=np.array(video_pred)])

# def diff(video_labels,video_pred):
# length=np.arange(len(video_labels))
# index=length[np.array(video_labels)!=np.array(video_pred)]
# data_labels = [video_labels[i] for i in index]
# data_pred = [video_pred[i] for i in index]
# print('The total different num is:',len(index))
# print('The index of differ :',index )
# print('The differ of video_labels:',data_labels)
# print('The differ of video_pred:', data_pred)

# diff(video_labels,video_pred)

# print('----'*30)

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
# print(cls_cnt, cls_hit)
cls_acc = cls_hit / cls_cnt

print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:
    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e: i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)



