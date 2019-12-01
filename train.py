import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from dataloader import DeepFakeDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument("--gpus", type=str,
                    help="Pass in like 1,2,3...")

def train(args):
    path = '/shared/gefenkohavi/data/fb_dfd/'
    
    model_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    
    trainset = DeepFakeDataset(path=path, split='train', oversample=True, transform=model_transforms)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = DeepFakeDataset(path=path, split='test', oversample=True, transform=model_transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Dropout(), nn.Linear(in_features=512, out_features=1, bias=True))
    model = model.cuda()
    
    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    running_loss = []
    for epoch in tqdm(range(args.epochs)):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.type(torch.FloatTensor).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(loss.item())
            if len(running_loss) > 100:
                running_loss.pop(0)
                
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[Epoch %d, Iter %5d] loss: %.3f' %
                      (epoch + 1, i + 1, sum(running_loss) / len(running_loss)))
                #print([int(ii) for ii in torch.round(torch.sigmoid(outputs.cpu())).detach().numpy()])
                #print([int(ii) for ii in labels.cpu().detach().numpy()])
                running_loss = []
        torch.save(model, 'resnetmodel/epoch_'+str(i)+'_model_resnet18.pth')
print('Finished Training')




if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
