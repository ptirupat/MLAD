import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numbers
import cv2
import random
import os
#import skvideo.io
import pickle
import h5py as h5
import math
import numpy as np
from configuration import build_config
import parameters as params
import time
from tqdm import tqdm


def filter_none(batch, max_len, varied_length=False):
    if max_len == -1:
        max_len = 0
        for item in batch:
            if item[0].shape[0] > max_len:
                max_len = item[0].shape[0]

    clips, labels = [], []
    for item in batch:
        if item[0] is not None and item[1] is not None:
            clip = torch.zeros(max_len, item[0].shape[1])
            label = torch.zeros(max_len, item[1].shape[1])
            clip[:item[0].shape[0]] = item[0]
            label[:item[1].shape[0]] = item[1]
            clips.append(clip)
            labels.append(label)
    clips, labels = torch.stack(clips), torch.stack(labels)
    if varied_length:
        length = random.choice(range(32, max_len, 16))
        clips, labels = clips[:,0:length,:], labels[:,0:length,:]
    return clips, labels


class Dataset(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_clips, skip, shuffle=True, add_background=False):
        self.data_split = data_split
        self.num_classes = cfg.num_classes
        self.annotations = json.load(open(cfg.annotations_file, 'r'))
        assert data_split in ['training', 'testing']
        assert input_type in ['rgb', 'flow', 'combined']
        if data_split == 'training':
            if input_type == 'rgb':
                self.features_file = cfg.rgb_train_file
            elif input_type == 'flow':
                self.features_file = cfg.flow_train_file
            elif input_type == 'combined':
                self.features_file = cfg.combined_train_file
            self.videos = [line.rstrip().replace('.txt', '') for line in open(cfg.train_list, 'r').readlines()]
        else:
            if input_type == 'rgb':
                self.features_file = cfg.rgb_test_file
            elif input_type == 'flow':
                self.features_file = cfg.flow_test_file
            elif input_type == 'combined':
                self.features_file = cfg.combined_test_file
            self.videos = [line.rstrip().replace('.txt', '') for line in open(cfg.test_list, 'r').readlines()]
        assert os.path.exists(self.features_file)
        if shuffle:
            random.shuffle(self.videos)
        len_data = int(len(self.videos) * data_percentage)
        self.videos = self.videos[0:len_data]
        self.data = None
        self.num_clips = num_clips
        self.skip = skip
        self.add_background = add_background

    def __len__(self):
        return len(self.videos)

    def build_labels(self, video_id, num_features, num_classes=65):
        labels = np.zeros((num_features, num_classes), np.float32)
        fps = num_features/self.annotations[video_id]['duration']
        for annotation in self.annotations[video_id]['actions']:
            for fr in range(0, num_features, 1):
                if fr/fps >= annotation[1] and fr/fps <= annotation[2]:
                    labels[fr, annotation[0] - 1] = 1   # will make the first class to be the last for datasets other than Multi-Thumos #          
        if self.add_background == True:
            new_labels = np.zeros((num_features, num_classes + 1))
            for i, label in enumerate(labels):
                new_labels[i,0:-1] = label
                if np.max(label) == 0:
                    new_labels[i,-1] = 1
            labels = new_labels
        return labels


    def __getitem__(self, index):
        if self.data is None:
            self.data = h5.File(self.features_file, 'r')
        video_id = self.videos[index]
        assert video_id in self.data.keys()
        features = self.data[video_id]
        labels = self.build_labels(video_id, len(features), self.num_classes)
        skip = random.choice(self.skip)
        num_clips = self.num_clips * (skip + 1) if skip > 0 else self.num_clips
        while num_clips >= len(features):
            skip = skip - 1
            if skip <= 0:
                num_clips = self.num_clips
                break
            else:
                num_clips = self.num_clips * (skip + 1) if skip > 0 else self.num_clips
        if len(features) > num_clips and num_clips > 0:
            random_index = random.choice(range(0, len(features) - num_clips))
            features = features[random_index : random_index + num_clips : skip + 1]
            labels = labels[random_index : random_index + num_clips : skip + 1]
        features = np.array(features)
        labels = np.array(labels)
        features = torch.from_numpy(features).to(torch.float)
        labels = torch.from_numpy(labels).squeeze()
        assert len(features) == len(labels)
        return features, labels

    
if __name__ == '__main__':
    shuffle = False
    cfg = build_config('charades')
    data_generator = Dataset(cfg, 'combined', 'training', 1.0, 128, skip=[0], shuffle=shuffle, add_background=False)
    dataloader = DataLoader(data_generator, batch_size=64, shuffle=shuffle, num_workers=0, collate_fn=lambda b : filter_none(b, 128, varied_length=False))
    start = time.time()
    counts = []
    class_counts = np.zeros((cfg.num_classes))
    for epoch in tqdm(range(0, 10)):
        for i, (clips, targets) in enumerate(dataloader):
            clips = clips.data.numpy()
            targets = targets.data.numpy()
            print(clips.shape, targets.shape)
    print("time taken : ", time.time() - start) 
