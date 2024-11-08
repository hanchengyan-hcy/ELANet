import os
import cv2
import torch
import random
import torch.utils.data as data

from option import args


class MEFdataset(data.Dataset):
    def __init__(self, transform):
        super(MEFdataset, self).__init__()
        self.dir_prefix = args.dir_train
        self.over = os.listdir(self.dir_prefix + 'over/')
        self.under = os.listdir(self.dir_prefix + 'under/')

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.over) == len(self.under)
        return len(self.over)

    def __getitem__(self, idx):
        over = cv2.imread(self.dir_prefix + 'over/' + self.over[idx], cv2.IMREAD_GRAYSCALE)
        # over = cv2.resize(over, (640, 512))
        under = cv2.imread(self.dir_prefix + 'under/' + self.under[idx], cv2.IMREAD_GRAYSCALE)
        # under = cv2.resize(under, (640, 512))

        if self.transform:
            over = self.transform(over)
            under = self.transform(under)

        return over, under


class TestData(data.Dataset):
    def __init__(self, transform):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test
        self.over_dir = os.listdir(self.dir_prefix + 'over/')
        self.under_dir = os.listdir(self.dir_prefix + 'under/')

    def __getitem__(self, idx):
        over = cv2.imread(self.dir_prefix + 'over/' + self.over_dir[idx], cv2.IMREAD_GRAYSCALE)
        under = cv2.imread(self.dir_prefix + 'under/' + self.under_dir[idx], cv2.IMREAD_GRAYSCALE)


        if self.transform:
            over_img = self.transform(over)
            under_img = self.transform(under)

        img_stack = torch.stack((over_img, under_img), 0)
        return img_stack

    def __len__(self):
        assert len(self.over_dir) == len(self.under_dir)
        return len(self.over_dir)