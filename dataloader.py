import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils

class DeepFakeDataset(Dataset):
    """Face Landmarks dataset."""
    
    methods = ['original_videos', 'method_A']
    label2int = {'real': 1, 'fake': 0}

    def __init__(self, path, split='train', oversample=True, transform=None):
        """
        Args:
            path (string): Path to the base of the facebook dataset
            oversample (string): Make sure classes are balanced when loading the data
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.path = path
        self.split = split
        self.oversample = oversample
        self.transform = transform
        
        with open(path + '/dataset.json') as f:
            self.meta = json.load(f)
            
        self.data = []
        for method in self.methods:
            self.crawl_dir(self.path, method)

    def crawl_dir(self, path, method):
        print('Crawling', method)
        for root, subdirs, files in tqdm(os.walk(path+'/images/'+method+'/')):
            if len(files) == 0: continue
            idx = root.index(method+'/')
            video_pth = root[idx:]+'.mp4'
            if video_pth in self.meta:
                info = self.meta[video_pth]
                if info['set'] == self.split:
                    files = [{'pth': '/'.join([root, f]), 'label': info['label']} for f in files]
                    self.data.extend(files)
            else:
                print(video_pth, 'not found!')
                1/0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]
        img = Image.open(info['pth'])
        label = self.label2int[info['label']]
                
        if self.transform:
            img = self.transform(img)

        return img, label
