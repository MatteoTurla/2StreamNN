import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision
import os
import time
import copy
import matplotlib.pyplot as plt
import sys
import numpy as np

from i3d.i3dpt import I3D
from stgcn.st_gcn import Model
from TwoStreamNN import TwoStreamNN
from myDataset import Dataset

for cluster in ['alerting','basic','daily_life']:
    nframes = 32
    downsample_pose = 1
    downsample_video = 2

    batch_size = 16

    dist_dir = f'stats_mean/stgcn_{cluster}_logits_1'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    readme_path = dist_dir+"/readme.txt"
    confusion_matrix_path = dist_dir+"/confusion_matrix.txt"

    dataset = Dataset('/home/Dataset/aggregorio_skeletons_numpy/{}/test'.format(cluster), 
                            '/home/Dataset/aggregorio_videos_pytorch_boxcrop/{}/test'.format(cluster),
                            n_frames=nframes, downsample_pose=downsample_pose, downsample_video=downsample_video, padding=False, balance=False)
    loader = loader_train = data.DataLoader(
                                    dataset,
                                    batch_size=batch_size,
                                	shuffle=False,
                                	pin_memory=True,
                                    num_workers = 8
                                )

    nclasses = len(dataset.classes)

    print('loading i3d')
    model_i3d = I3D(num_classes=nclasses, modality='rgb')
    model_i3d.load_state_dict(torch.load('i3d/best_model/{}.pt'.format(cluster)))
    model_i3d.cuda()

    print('loading stgcn')
    model_args = {
      'in_channels': 3,
      'num_class': nclasses,
      'edge_importance_weighting': True,
      'graph_args': {
        'layout': 'openpose',
        'strategy': 'spatial'
      }
    }
    model_stgcn = Model(model_args['in_channels'], model_args['num_class'],
                    model_args['graph_args'], model_args['edge_importance_weighting'])
    model_stgcn.load_state_dict(torch.load('stgcn/best_model/{}.pt'.format(cluster)))
    model_stgcn.cuda()

    model_2stream = TwoStreamNN(model_i3d, model_stgcn, nclasses)
    model_2stream = model_2stream.cuda()

    #matrice confusione
    print("Calc confusion matrix")
    model_2stream.eval()
    conf_matrix = torch.zeros(nclasses, nclasses)
    conf_matrix_prob = torch.zeros(nclasses, nclasses)
    with torch.no_grad():
        for poses, videos, labels in loader:
            print(poses.shape, videos.shape)
            poses = poses.cuda()
            videos = videos.cuda()
            labels = labels.cuda()

            y_, logits_ = model_2stream.stgcn(poses)

            _prob, y_label_ = torch.max(y_, 1)

            for i in range(len(labels)):
                conf_matrix[labels[i].item(), y_label_[i].item()] += 1

            for i in range(len(labels)):
                conf_matrix_prob[labels[i].item(), y_label_[i].item()] += _prob[i]

    _bins = dataset.bincount()
    bins = [_bin.item() for _bin in _bins]
    for i in range(nclasses):
        for j in range(nclasses):
            conf_matrix[i][j] = conf_matrix[i][j]/bins[i]

    for i in range(nclasses):
        for j in range(nclasses):
            conf_matrix_prob[i][j] = conf_matrix_prob[i][j]/bins[i]

    mean_tot = 0
    for i in range(nclasses):
        mean_tot += conf_matrix[i][i].item()
    mean_tot /= nclasses

    #ßaving some stats

    orig_stdout = sys.stdout
    with open(readme_path, 'w+') as f:
        sys.stdout = f
        dataset.print()
    sys.stdout = orig_stdout


    orig_stdout = sys.stdout
    with open(confusion_matrix_path, 'w+') as f:
        sys.stdout = f
        print(conf_matrix)
        print()
        print(conf_matrix_prob)
        print()
        print('accuracy:', mean_tot)
    sys.stdout = orig_stdout

