import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from dataloader import DeepFakeDataset
from models import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument("--gpus", type=str,
                    help="Pass in like 1,2,3...")
parser.add_argument("--model", type=str, required=True)
def train(args):
    path = '/shared/gefenkohavi/data/fb_dfd/'
    
    model_transforms = transforms.Compose([
#        transforms.Resize((299, 299)),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    
#     trainset = DeepFakeDataset(path=path, split='train', oversample=True, transform=model_transforms)
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = DeepFakeDataset(path=path, split='test', oversample=True, transform=model_transforms)
    weights = testset.weights  
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=sampler)
    

#     model = models.resnet18(pretrained=True)
#     model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    #model = resnet._resnet('resnet18', resnet.BasicBlock, [1, 2, 2, 2], pretrained=False, progress=False)
    #model.fc = nn.Sequential(nn.Dropout(), nn.Linear(in_features=512, out_features=1, bias=True))
    model = torch.load(args.model)
    model = model.cuda()
    model = model.eval()
    
    criterion = nn.BCEWithLogitsLoss()

    running_loss = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.type(torch.FloatTensor).cuda()


            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            # print statistics
            running_loss.append(loss.item())
            if len(running_loss) > 100:
                running_loss.pop(0)
                
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[Iter %5d] loss: %.3f' %
                      (i + 1, sum(running_loss) / len(running_loss)))
                a = [int(ii) for ii in torch.round(torch.sigmoid(outputs.cpu())).detach().numpy()]
                b = [int(ii) for ii in labels.cpu().detach().numpy()]

                print("Acc:", sum(aa==bb for aa, bb in zip(a, b))/len(a))
                running_loss = []
print('Finished Training')




if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
