# -*- coding:utf-8 -*-

import torch.utils.data as data
import torch
import torch.nn as nn
import torchvision

from PIL import Image
import numpy as np
import random
import os
import os.path
import scipy.io as scio
#
# Return random integers from low (inclusive) to high (exclusive)
#
from numpy.random import randint
from transforms import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_model(hparams_model, pretrained=True):
    if hparams_model == 'vanilla':
        model = vanilla_models.__dict__['VanillaCNN']()
    else:
        model = torchvision.models.__dict__[hparams_model](
            pretrained=pretrained)

    # with open("/data/cuiyujie/network/vit-pytorch-main/result/original_{}.txt".format(hparams_model), 'w') as w:
    #     w.write("{}\n".format(model))

    # Replace fc to identity, treat it as backbone
    dim = model.fc.in_features
    model.fc = Identity()

    # with open("/data/cuiyujie/network/vit-pytorch-main/result/modified_{}.txt".format(hparams_model), 'w') as w:
    #     w.write("{}\n".format(model))

    return model


#
# used for return image folders
#
def default_collate(insts):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    '''
    with open("/data/cuiyujie/network/vit-pytorch-main/result/tangent_default_collate2.txt", 'w') as w:
        for i, inst in enumerate(insts):
            w.write("{}\n{}\n{}\n\n".format(inst[0].shape, inst[1], inst[2]))
    '''

    #
    # batch_input.shape is torch.Size([3, 12, 256, 256])
    #
    batch_input = torch.stack([ins[0] for ins in insts], dim=0)
    batch_label = torch.tensor(np.array([ins[1] for ins in insts]))

    batch_folders = list()
    for ins in insts:
        # print(ins[2])
        batch_folders.append(ins[2])

    # print("result_folders:\n{}".format(result_folders))
    # for result in result_folders:
    #     print("result:\n{}".format(result))
    #     print("length:\n{}\n".format(len(result)))
    # print("length:\n{}\n".format(len(result_folders)))

    # batch_folders = [insts[0][2].append(it[2] for it in insts[1:])]
    # print(batch_folders)

    # batch_folders =

    # print("batch_input: {}".format(batch_input.shape))
    # print("batch_label: {}".format(batch_label))

    # batch_folder =

    # img_path = [''.join(ins[2]) for ins in insts]

    # with open("/home/cuiyujie/tsn-pytorch-master/result/output/img_path_in_collate.txt", 'w') as w:
    #     w.write("{}\n".format(img_path))

    # with open(
    #         "/home/cuiyujie/tsn-pytorch-master/result/output/spherePHD_default_collate_insts2.txt",
    #         'a') as w:
    #     for ins in insts:
    #         w.write("{}\n{}\n\n{}\n{}\n\n".format(batch_input.shape,
    #                                               batch_input,
    #                                               batch_label.shape,
    #                                               batch_label))

    # return
    return batch_input, batch_label, batch_folders


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


#
# clip_length means num_segments
# tangent_nums means number of tangent images
#
class TangentDataSet(data.Dataset):
    def __init__(self, list_file, num_frames=10,
                 tangent_nums=20, num_faces=3, frame_size=(256, 256),
                 image_folder_tmpl='img_{:05d}', image_tmpl='image{:06d}.png',
                 transform=None, random_sample=True, random_shift=True, test_mode=False):
        # self.root_path = root_path
        self.list_file = list_file
        self.num_frames = num_frames  # the segmentation of video num
        self.tangent_nums = tangent_nums  # sample tangent faces num
        self.num_faces = num_faces
        self.frame_size = frame_size
        self.image_folder_tmpl = image_folder_tmpl
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_sample = random_sample
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in
                           open(self.list_file)]

    #
    # sample indices of consecutive num of tangent faces
    # 返回self.num_segments个在0到self.tangent_nums范围内的indices
    #
    def _tangent_indices(self):
        # return randint(self.tangent_nums, self.num_segments).sort()  # sequential
        if self.num_faces == 20:
            if self.random_sample:
                offsets = list(range(20))
                random.shuffle(offsets)  # random shuffle sampled list            
            else:
                offsets = range(20)  # sequential sample list
        else:
            tick = randint(self.tangent_nums - self.num_faces + 1)
            #
            # 采样连续的不同面，捕捉空间信息
            #
            offsets = np.array(
                [tick + x for x in range(self.num_faces)]
            )
        return offsets

    #
    # load self.num_faces consecutive tangent images
    #
    def _load_images(self, directory, idx, tangent_ids):
        # video directory
        # directory = os.path.join(self.root_path, directory)
        # frame folders
        image_folder = os.path.join(directory, self.image_folder_tmpl.format(idx))
        # imgs = list()

        # sample images with consecutive tangent_indicies in image_folder
        imgs = [Image.open(os.path.join(image_folder, self.image_tmpl.format(idx))).convert('RGB') for idx in tangent_ids]
        # for idx in tangent_ids:
        #     try:
        #         img_path = os.path.join(image_folder, self.image_tmpl.format(idx))
        #         imgs.append(Image.open(img_path).convert('RGB'))
        #     except IOError:
        #         with open("/data/cuiyujie/data/EgoK360_list/truncated_imgs.txt", 'a') as w:
        #             w.write("{}\n".format(img_path))
        return imgs
        
    #
    # sample indices of image_folders
    #
    def _sample_indices(self, record):
        if not self.test_mode and self.random_shift:
            #
            # average_duration means num of frames in each clip
            #
            average_duration = record.num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.sort(
                    np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments))
            else:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
        # in test mode
        else:
            tick = record.num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        indices = self._sample_indices(record)

        return self.get(record, indices)

    #
    # indices mean image folder ids
    # tangent_indices mean tangent face ids
    #
    def get(self, record, indices):
        images = list()
        #
        # sample same tangent faces in different images
        #
        tangent_ids = self._tangent_indices()
        # 
        # 采样所有的tangnet_images
        # tangent_ids = range(self.tangent_nums)

        for index in indices:
            # imgs, image_folder = self._load_images(record.path, int(index))
            #
            # consecutive tangent frames in different image folders
            #
            imgs = self._load_images(record.path, int(index), tangent_ids)
            images.extend(imgs)
            # image_folders.append(image_folder)

        process_data = self.transform(images)  # process_data.shape is [3, 10, 224, 224], [channel, frame_num, width, height]
        # image_folders = '\n'.join(image_folders)
        # folders = tuple()

        # return process_data, record.label, image_folders
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


def get_augmentation(input_size):
    return torchvision.transforms.Compose(
        [GroupMultiScaleCrop(input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])


if __name__ == '__main__':
    tangent_train_list = "/data/cuiyujie/data/EgoK360_list/egok360_tangent8_test5_train_shuffle.txt"
    num_segments = 3
    batch_size = 8
    workers = 4
    tangent_input_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    hparams_model = 'resnet18'
    model = get_model(hparams_model, pretrained=True)

    tangent_train_augmentation = get_augmentation(tangent_input_size)

    tangent_dataset = TangentDataSet(tangent_train_list, num_segments=num_segments,
                                     transform=torchvision.transforms.Compose([
                                         tangent_train_augmentation,
                                        #  ToNumpyNDArray(),
                                         Stack(roll=False),
                                         ToTorchFormatTensor(div=True),
                                         GroupNormalize(input_mean, input_std)
                                     ]))
    # print(len(dataset)) .

    tangent_train_loader = torch.utils.data.DataLoader(
        tangent_dataset,
        batch_size=batch_size, shuffle=True,
        # collate_fn=default_collate,
        num_workers=workers, pin_memory=True)

    #
    # in tangent dataset:
    # batch, channel, num_frames*num_tangent_faces, image_size
    # batch: number of video clips
    # channel: rgb
    # num_frames: 3, num_tangent_faces: 3 (sample 3 frames in a video and sample 3 faces in icosahedron faces)
    # image_size: [H, W], H and W are cropped image size
    #
    # input.shape is torch.Size([8, 3, 9, 224, 224])
    # target.shape is torch.Size([8])
    #
    #
    # in i3d dataset:
    #
    # input.shape is torch.Size([8, 3, 4, 224, 224])
    # target.shape is torch.Size([8])
    #
    # for i, (input, target, image_folders) in enumerate(tangent_train_loader):
    for i, (input, target) in enumerate(tangent_train_loader):
        
        print("input.shape: ", input.shape)
        print("target.shape: ", target.shape)
        
        #
        # input.shape[0] is batch_size, which is 8
        # input.shape[2] is frame_size, which is 12(4 clip_length * 3 segment_nums)
        #
        '''
        input = input.reshape(input.shape[0] * input.shape[2], input.shape[1], input.shape[3], input.shape[4])
        output = model(input)
        print("reshaped input: {}".format(input.shape))
        # print("output: {}".format(output.shape))
        # print("output.shape[0]: {}".format(output.shape[0]))
        # print("output.shape[-1]: {}".format(output.shape[-1]))

        output = output.reshape([batch_size, output.shape[0] // batch_size, output.shape[-1]])
        print("reshaped output: {}".format(output.shape))
        '''
        # print(i)
        # pass
        break
