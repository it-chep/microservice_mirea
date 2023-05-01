import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision as tv
import pandas as pd

import numpy as np
from tqdm.autonotebook import tqdm
import torchvision.models as models

csv_path = 'fer2013.csv'


class FER2013Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, train=True):
        self.train = train
        self.trans = tv.transforms.Compose([
            tv.transforms.Resize((128, 128)),
            tv.transforms.ToTensor(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomRotation(45)
        ])

        if train:
            self.data = pd.read_csv(csv_file, nrows=27000)
        else:
            self.data = pd.read_csv(csv_file, skiprows=range(1, 27000))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, ):

        pixels = self.data.loc[idx, 'pixels']
        pixels = [int(pixel) for pixel in pixels.split()]
        pixels = np.array(pixels).reshape((48, 48)).astype(np.uint8)
        pixels = Image.fromarray(pixels)
        label = self.data.loc[idx, 'emotion']

        return {'pixels': self.trans(pixels), 'label': label}


class Net(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU(0.01)

        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 7)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)

        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.maxpool(out)
        out = self.act(out)
        out = self.adaptive(out)

        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out


# model = Net()

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 7)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

loss_function = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 50


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy()
    return answer.mean()


def work_with_ai(loss_val, acc_val, train, dataloader):

    for batch in (pbar := tqdm(dataloader)):
        pixels, label = batch['pixels'], batch['label']
        pixels = pixels.float().squeeze().unsqueeze(1)
        label = label.long()

        optimizer.zero_grad()

        pred = model(pixels)
        loss = loss_function(pred, label)
        if train:
            scaler.scale(loss).backward()
            loss_item = loss.item()
            loss_val += loss_item

            scaler.step(optimizer)
            scaler.update()
        else:
            loss_item = loss.item()
            loss_val += loss_item

        acc_current = accuracy(pred.cpu().float(), label.cpu().float())
        acc_val += acc_current
    pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')


def validate(loss_val, acc_val):
    model.eval()

    dataset = FER2013Dataset(csv_file=csv_path, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                             num_workers=4, drop_last=True)
    with torch.no_grad():
        work_with_ai(loss_val, acc_val, False, dataloader)
        print('[+]ТЕСТЫ[+]')
        print(loss_val / len(dataloader))
        print(acc_val / len(dataloader))


def training_model():
    dataset = FER2013Dataset(csv_file=csv_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                             num_workers=4, drop_last=True)
    for epoch in range(epochs):
        loss_val = 0.0
        acc_val = 0.0

        work_with_ai(loss_val, acc_val, True, dataloader)

        print('[+]', epoch, 'номер эпохи[+]')
        print(loss_val / len(dataloader))
        print(acc_val / len(dataloader))
        validate(0.0, 0.0, )

        torch.save(model.state_dict(), f'saved_epochs/emoji_last1_model_epoch_{epoch + 1}.pth')


training_model()
