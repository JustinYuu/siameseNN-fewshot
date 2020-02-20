import torch
import math
import torch.optim as optim
import torch.nn as nn
from dataloader import *
from net import SiameseNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, mean=0, std=0.2)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.BATCH_SIZE = 128
        self.EPOCH = args.epoch
        self.way = args.way
        self.times = args.times
        self.lr = 0.0001

        train_transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.02, 0.02), scale=(0.8, 1.2), shear=0.3 * 180/math.pi),
            transforms.ToTensor()
        ])

        train_data = OmniglotTrain('data/images_background', transform=train_transform)
        test_data = OmniglotTest('data/images_evaluation', transform=transforms.ToTensor(), times=self.times, way=self.way)
        self.train_loader = DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(test_data, batch_size=self.way, shuffle=False, num_workers=1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.net = SiameseNN()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.apply(weight_init)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)

    def train(self):
        best_acc = 0
        for epoch in range(self.EPOCH):
            self.net.train()
            sum_loss = 0.0
            for i, data in enumerate(self.train_loader):
                input1, input2, labels = data
                if torch.cuda.is_available():
                    input1, input2, labels = input1.cuda(), input2.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.net(input1, input2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                if i % 128 == 127:
                    print('[%d %d] loss:%.03f' %
                          (epoch + 1, i + 1, sum_loss/128))
                    sum_loss = 0.0
            self.scheduler.step()
            acc = self.test()
            print('Epoch%d acc: %.03f' % (epoch+1, acc))
            if acc > best_acc:
                best_acc = acc
                state = {
                    'net': self.net.state_dict(),
                    'acc': best_acc,
                    'epoch': self.EPOCH
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, 'checkpoint/checkpoint.pth')
        print("best acc is:%.03f" % best_acc)

    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (test1, test2) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    test1, test2 = test1.cuda(), test2.cuda()
                outputs_test = self.net(test1, test2)
                outputs_test = outputs_test.data.cpu().numpy()
                predicted = np.argmax(outputs_test)
                if predicted == 0 and outputs_test.std() > 0.01:   # To prevent from all elements are almost same
                    correct += 1
                total += 1
        acc = correct / total
        print("correct={}, total={}".format(correct, total))
        return acc
