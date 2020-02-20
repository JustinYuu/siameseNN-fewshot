import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy.random as random


class OmniglotTrain(Dataset):
    def __init__(self, path, transform=None):
        super(OmniglotTrain, self).__init__()
        self.path = path
        self.transform = transform
        data = {}
        index = 0
        total = 0
        for root_path in os.listdir(path):
            for sub_path in os.listdir(os.path.join(path, root_path)):
                data[index] = []
                for sub_sub_path in os.listdir(os.path.join(path, root_path, sub_path)):
                    img_path = os.path.join(path, root_path, sub_path, sub_sub_path)
                    data[index].append(Image.open(img_path).convert('L'))
                    total += 1
                index += 1
        self.data = data
        self.num_classes = index
        self.length = total

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        flag = np.random.randint(0, 1)
        if flag == 1:  # same class
            category = random.randint(0, self.num_classes - 1)
            imgnum1 = random.randint(0, len(self.data[category]) - 1)
            imgnum2 = random.randint(0, len(self.data[category]) - 1)
            img1 = self.data[category][imgnum1]
            img2 = self.data[category][imgnum2]
        else:  # distinct class
            category1 = random.randint(0, self.num_classes - 1)
            category2 = random.randint(0, self.num_classes - 1)
            while category1 == category2:
                category2 = random.randint(0, self.num_classes - 1)
            imgnum1 = random.randint(0, len(self.data[category1]) - 1)
            imgnum2 = random.randint(0, len(self.data[category2]) - 1)
            img1 = self.data[category1][imgnum1]
            img2 = self.data[category2][imgnum2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([flag], dtype=np.float32))


class OmniglotTest(Dataset):
    def __init__(self, path, transform=None, times=200, way=20):
        super(OmniglotTest, self).__init__()
        self.path = path
        self.transform = transform
        self.times = times
        self.way = way
        self.target = None
        self.target_category = None
        data = {}
        index = 0
        for root_path in os.listdir(path):
            for sub_path in os.listdir(os.path.join(path, root_path)):
                data[index] = []
                for sub_sub_path in os.listdir(os.path.join(path, root_path, sub_path)):
                    img_path = os.path.join(path, root_path, sub_path, sub_sub_path)
                    data[index].append(Image.open(img_path).convert('L'))
                index += 1
        self.data = data
        self.num_classes = index

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        isFirst = index % self.way
        if isFirst == 0:  # put image from the same class in front
            category = random.randint(0, self.num_classes - 1)
            imgnum1 = random.randint(0, len(self.data[category]) - 1)
            imgnum2 = random.randint(0, len(self.data[category]) - 1)
            img1 = self.data[category][imgnum1]
            img2 = self.data[category][imgnum2]
            self.target = img1
            self.target_category = category
        else:  # put images from distinct classes behind
            category = random.randint(0, self.num_classes - 1)
            while self.target_category == category:
                category = random.randint(0, self.num_classes - 1)
            imgnum = random.randint(0, len(self.data[category]) - 1)
            img1 = self.target
            img2 = self.data[category][imgnum]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        #print('img1 category:{}, img2 category:{}'.format(self.target_category, category))
        return img1, img2
